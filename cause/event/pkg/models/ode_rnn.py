"""Recurrent Mark Density Estimator
"""
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from ..explain.integrated_gradient import batch_integrated_gradient
from ..utils.misc import AverageMeter
from ..utils.torch import ResidualLayer, generate_sequence_mask, set_eval_mode
from .func_basis import Normal, Unity
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint
from .rnn_bak import ExplainableRecurrentPointProcess



class ScaleNet(nn.Module):
    def __init__(self, scale, func, t0=None, t1=None):
        super(ScaleNet, self).__init__()
        # if not isinstance(func, torch.nn.Module):
        #     raise ValueError('func must be a nn.Module')
        self.scale = scale.detach()
        self.func = func
        self.t0 = t0
        self.t1 = t1

    def __call__(self, t, z):
        gradient = self.func(self.t0 + t * self.scale, z)
        return gradient * self.scale

class StableODE(nn.Module):
    def __init__(self, stable=True, embedding_dim=64):
        super(StableODE, self).__init__()
        if stable:
            self.ode = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh()
            )
        else:
            self.ode = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, embedding_dim),
            )
        self.beta = 1
        self.stable = stable
    def __call__(self, t, z):
        if self.stable:
            gradient = self.ode(z) - z
        else:
            gradient = self.ode(z)
        return gradient * self.beta


# class ExplainableRecurrentPointProcess(nn.Module):
class ODERecurrentPointProcess(nn.Module):
    def __init__(
        self,
        n_types: int,
        embedding_dim: int = 32,
        hidden_size: int = 32,
        dropout: float = 0.0,
        rtol=1e-7,
        atol=1e-9,
        rnn: str = "GRU",
        method='rk4',
        **kwargs,
    ):
        super().__init__()
        self.n_types = n_types

        # self.embed = nn.Linear(n_types, embedding_dim, bias=False)
        self.embed = nn.Embedding.from_pretrained(torch.zeros(n_types, embedding_dim)).requires_grad_(True)
        self.dropout = nn.Dropout(p=dropout)
        # self.rnn = nn.GRUCell(
        #     input_size=embedding_dim,
        #     hidden_size=hidden_size,
        # )
        self.rnn = nn.GRU(
            embedding_dim + 1, hidden_size, 1, batch_first=True
        )

        # self.stable_ode = StableODE(stable=True, embedding_dim=embedding_dim)
        self.stable_ode = StableODE(stable=False, embedding_dim=embedding_dim)

        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.dropout = nn.Dropout(p=dropout)

        # self.q_net = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size, n_types),
        # )


        self.q_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        # self.psi_net = ResidualLayer(in_features=hidden_size, out_features=hidden_size, hidden_size=hidden_size)
        self.psi_net = nn.Linear(hidden_size, hidden_size)


    def all_type_log_intensities(self, x):
        q = self.q_net(x)
        # psi = self.psi_net(self.embed.weight.detach()).permute(1, 0).contiguous()
        psi = self.psi_net(self.embed.weight).permute(1, 0).contiguous()
        log_intensities = q @ psi
        return log_intensities

        # log_intensities = self.q_net(x)
        # return log_intensities


    def forward(
            self, event_seqs, onehot=False, need_h_positive=False, target_type=-1
    ):
        """[summary]

        Args:
          event_seqs (Tensor): shape=[batch_size, T, 2]
            or [batch_size, T, 1 + n_types]. The last dimension
            denotes the timestamp and the type of an event, respectively.

          onehot (bool): whether the event types are represented by one-hot
            vetors.

         need_cumulative (bool): whether to return the cumulative.

          target_type (int): whether to only predict for a specific type

        Returns:
           log_intensities (Tensor): shape=[batch_size, T, n_types],
             log conditional intensities evaluated at each event for each type
             (i.e. starting at t1).
           weights (Tensor, optional): shape=[batch_size, T, n_types, n_bases],
             basis weights intensities evaluated at each previous event (i.e.,
             tarting at t0). Returned only when `need_weights` is `True`.

        """
        assert event_seqs.size(-1) == 1 + (
            self.n_types if onehot else 1
        ), event_seqs.size()

        # When onehot is True, need_cumulative must be False
        assert not onehot or not need_h_positive

        batch_size, T = event_seqs.size()[:2]

        # (t0=0, t1, t2, ..., t_n)
        ts = F.pad(event_seqs[:, :, 0], (1, 0))
        # (0, t1 - t0, ..., t_{n} - t_{n - 1})
        # dt = F.pad(ts[:, 1:] - ts[:, :-1], (1, 0))
        dt = ts[:, 1:] - ts[:, :-1]
        # (0, t1 - t0, ..., t_{n - 1} - t_{n - 2})
        temp_feat = dt.unsqueeze(-1)

        # (0, z_1, ..., z_{n - 1})
        if onehot:
            type_feat = event_seqs[:, :-1, 1:].float() @ self.embed.weight
        else:
            type_feat = self.embed.weight[event_seqs[:, :-1, 1].long()]
        type_feat = F.pad(type_feat, (0, 0, 1, 0))

        type_feat = torch.cat([F.pad(temp_feat, (0, 0, 1, 0))[:, :-1], type_feat], dim=-1)

        # feat = torch.cat([temp_feat, type_feat], dim=-1)
        # history_emb, *_ = self.rnn(feat)
        # history_emb = self.dropout(history_emb)

        # h_pos = torch.zeros((batch_size, self.rnn.hidden_size), device=event_seqs.device, dtype=event_seqs.dtype)
        # h_neg = torch.zeros((batch_size, self.rnn.hidden_size), device=event_seqs.device, dtype=event_seqs.dtype)
        h_neg = None
        h_positive_all = []
        h_negative_all = []

        for i in range(T):
            # h_pos = self.rnn(type_feat[:, i], h_neg)
            _, h_pos = self.rnn(type_feat[:, i:i+1], h_neg)
            h_pos = h_pos[0]
            h_positive_all.append(h_pos)
            scale_net = ScaleNet(
                scale=torch.clip(temp_feat[:,i], 1e-3, None),
                func=self.stable_ode,
                t0=ts[:, i],
                t1=ts[:, i+1],
            )
            ys = odeint(scale_net, h_pos, torch.Tensor([0.0, 1.0]).to(h_pos.device), rtol=self.rtol, atol=self.atol, method=self.method)
            h_neg = ys[-1]
            h_negative_all.append(h_neg)
            h_neg = h_neg.unsqueeze(dim=0)

        h_positive_all = self.dropout(torch.stack(h_positive_all, dim=1))
        h_negative_all = self.dropout(torch.stack(h_negative_all, dim=1))

        log_intensities = self.all_type_log_intensities(h_negative_all)

        # [B, T, 1, R]
        # basis_values = torch.cat(
        #     [basis.log_prob(dt[:, 1:, None]) for basis in self.bases], dim=2
        # ).unsqueeze(-2)

        if target_type >= 0:
            log_intensities = log_intensities[
                                :, :, target_type: target_type + 1
                                ]

        # log_intensities = (log_basis_weights + basis_values).logsumexp(dim=-1)
        # if need_weights:
        #     return log_intensities, log_basis_weights
        # else:
        if need_h_positive:
            return log_intensities, h_positive_all
        else:
            return log_intensities


    # def _eval_cumulants(self, batch, history_emb):
    #     """Evaluate the cumulants (i.e., integral of CIFs at each location)
    #     """
    #     ts = batch[:, :, 0]
    #     # (t1 - t0, ..., t_n - t_{n - 1})
    #     dt = (ts - F.pad(ts[:, :-1], (1, 0))).unsqueeze(-1)
    #     # [B, T, R]
    #     integrals = torch.cat(
    #         [
    #             basis.cdf(dt) - basis.cdf(torch.zeros_like(dt))
    #             for basis in self.bases
    #         ],
    #         dim=-1,
    #     )
    #     cumulants = integrals.unsqueeze(2).mul(log_basis_weights.exp()).sum(-1)
    #     return cumulants
    def _eval_cumulants(self, batch, h_positive_all):
        """Evaluate the cumulants (i.e., integral of CIFs at each location)
        """
        ts = batch[:, :, 0]
        # (t1 - t0, ..., t_n - t_{n - 1})
        ts = F.pad(ts, (1, 0))
        # dt = (ts - F.pad(ts[:, :-1], (1, 0))).unsqueeze(-1)
        dt = (ts[:, 1:] - ts[:, :-1]).unsqueeze(-1)

        def f(t, x_aug):
            x = x_aug[..., :-self.n_types]
            log_intensities = self.all_type_log_intensities(x)
            return torch.cat([self.stable_ode(t, x), torch.exp(log_intensities)], dim=-1)

        x_aug_t0 = torch.cat([
            h_positive_all,
            torch.zeros((*(h_positive_all.size()[:-1]), self.n_types), device=h_positive_all.device, dtype=h_positive_all.dtype)
        ], dim=-1)
        scale_net = ScaleNet(
            scale=torch.clip(dt, 1e-3, None),
            func=f,
            t0=ts[:, :-1].unsqueeze(-1),
            t1=ts[:, 1:].unsqueeze(-1),
        )

        xs = odeint(scale_net, x_aug_t0, torch.Tensor([0.0, 1.0]).to(h_positive_all.device), rtol=self.rtol, atol=self.atol,
                    method=self.method)
        cumulants = xs[-1, ..., -self.n_types:]

        return cumulants
    def _eval_nll(
            self, batch, log_intensities, cumulative, mask, debug=False
    ):

        loss_part1 = (
            -log_intensities.gather(dim=2, index=batch[:, :, 1:].long())
            .squeeze(-1)
            .masked_select(mask)
            .sum()
        )

        loss_part2 = (
            cumulative
            .sum(-1)
            .masked_select(mask)
            .sum()
        )
        if debug:
            return (
                (loss_part1 + loss_part2) / batch.size(0),
                loss_part1 / batch.size(0),
            )
        else:
            return (loss_part1 + loss_part2) / batch.size(0)

    def _eval_acc(self, batch, intensities, mask):
        types_pred = intensities.argmax(dim=-1).masked_select(mask)
        types_true = batch[:, :, 1].long().masked_select(mask)
        return (types_pred == types_true).float().mean()

    def train_epoch(
        self,
        train_dataloader,
        optim,
        valid_dataloader=None,
        device=None,
        **kwargs,
    ):
        self.train()

        train_metrics = defaultdict(AverageMeter)

        for batch in train_dataloader:
            if device:
                batch = batch.to(device)
            seq_length = (batch.abs().sum(-1) > 0).sum(-1)
            mask = generate_sequence_mask(seq_length)

            log_intensities, h_positive_all = self.forward(
                batch, need_h_positive=True
            )
            cumulative = self._eval_cumulants(batch, h_positive_all)
            nll = self._eval_nll(
                batch, log_intensities, cumulative, mask
            )
            loss = nll

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_metrics["loss"].update(loss, batch.size(0))
            train_metrics["nll"].update(nll, batch.size(0))
            train_metrics["acc"].update(
                self._eval_acc(batch, log_intensities, mask), seq_length.sum()
            )

        if valid_dataloader:
            valid_metrics = self.evaluate(valid_dataloader, device=device)
        else:
            valid_metrics = None

        return train_metrics, valid_metrics

    def evaluate(self, dataloader, device=None):
        metrics = defaultdict(AverageMeter)

        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                if device:
                    batch = batch.to(device)

                seq_length = (batch.abs().sum(-1) > 0).sum(-1)
                mask = generate_sequence_mask(seq_length)

                log_intensities, h_positive_all = self.forward(
                    batch, need_h_positive=True
                )
                cumulative = self._eval_cumulants(batch, h_positive_all)
                nll = self._eval_nll(
                    batch, log_intensities, cumulative, mask
                )

                metrics["nll"].update(nll, batch.size(0))
                metrics["acc"].update(
                    self._eval_acc(batch, log_intensities, mask),
                    seq_length.sum(),
                )

        return metrics


    def get_seq_contribution(
            self,
            batch,
            device=None,
            steps=50,
            occurred_type_only=False,
            **kwargs,
    ):
        def func(X):
            log_intensities = self.forward(
                X, onehot=True, need_cumulative=False
            )
            return log_intensities
            # log_intensities_p = torch.gather(
            #     log_intensities[:, position],
            #     dim=-1,
            #     index=torch.argmax(X[:, p, 1:], dim=-1)
            # )
            # return log_intensities_p

            # cumulants = self._eval_cumulants(X, log_basis_weights)
            # drop index=0 as it corresponds to (t_0, t_1)
            # return cumulants[:, 1:]

        set_eval_mode(self)
        # freeze the model parameters to reduce unnecessary backpropogation.
        for param in self.parameters():
            param.requires_grad_(False)

        # A = torch.zeros(self.n_types, self.n_types, device=device)
        # type_counts = torch.zeros(self.n_types, device=device).long()

        if device:
            batch = batch.to(device)

        batch_size, T = batch.size()[:2]
        seq_lengths = (batch.abs().sum(-1) > 0).sum(-1) # (batch_size, ) 每条数据的有效长度

        inputs = torch.cat(
            [
                batch[:, :, :1],
                F.one_hot(batch[:, :, 1].long(), self.n_types).float(),
            ],
            dim=-1,
        )
        baselines = F.pad(inputs[:, :, :1], (0, self.n_types))
        mask = generate_sequence_mask(seq_lengths - 1, device=device)

        # if occurred_type_only:
        #     occurred_types = set(
        #         batch[:, :, 1]
        #         .masked_select(generate_sequence_mask(seq_lengths))
        #         .long()
        #         .tolist()
        #     )
        # else:
        #     occurred_types = range(self.n_types)

        # event_scores = torch.zeros(
        #     self.n_types, batch_size, T - 1, device=device
        # )

        # event_scores[i,j,k]: 第i个样本中，第k个事件对第j个事件的contribution
        event_scores = torch.zeros(
            batch_size, T, T - 1, device=device
        )

        for pos in range(T):
            ig = batch_integrated_gradient(
                partial(func, position=pos),
                inputs,
                baselines=baselines,
                mask=mask[:, pos].unsqueeze(-1),
                steps=steps,
                idx=pos
            )
            event_scores[:, pos] = ig[:, :-1].sum(-1)

        return event_scores.detach().cpu()

        # for k in occurred_types:
        # #     ig = batch_integrated_gradient(
        # #         partial(func, position=k),
        # #         inputs,
        # #         baselines=baselines,
        # #         mask=mask.unsqueeze(-1),
        # #         steps=steps,
        # #     )
        # #     event_scores[k] = ig[:, :-1].sum(-1)
        #
        # # shape=[K, B, T - 1]
        # A.scatter_add_(
        #     1,
        #     index=batch[:, :-1, 1]
        #     .long()
        #     .view(1, -1)
        #     .expand(self.n_types, -1),
        #     src=event_scores.view(self.n_types, -1),
        #     )
        #
        # ks = (
        #     batch[:, :, 1]
        #     .long()
        #     .masked_select(generate_sequence_mask(seq_lengths))
        # )
        # type_counts.scatter_add_(0, index=ks, src=torch.ones_like(ks))
        #
        # # plus one to avoid divison by zero
        # A /= type_counts[None, :].float() + 1
        #
        # return A.detach().cpu()


