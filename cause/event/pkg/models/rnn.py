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


class EventSeqDataset(Dataset):
    """Construct a dataset for store event sequences.

    Args:
        event_seqs (list of list of 2-tuples):
    """

    def __init__(self, event_seqs, min_length=1, sort_by_length=False):

        self.min_length = min_length
        self._event_seqs = [
            torch.FloatTensor(np.array(seq).astype(float))
            for seq in event_seqs
            if len(seq) >= min_length
        ]
        if sort_by_length:
            self._event_seqs = sorted(self._event_seqs, key=lambda x: -len(x))

    def __len__(self):
        return len(self._event_seqs)

    def __getitem__(self, i):
        # TODO: can instead compute the elapsed time between events
        return self._event_seqs[i]

    @staticmethod
    def collate_fn(X):
        return nn.utils.rnn.pad_sequence(X, batch_first=True)


class EventSeqWithLabelDataset(Dataset):
    """Construct a dataset for store event sequences.

    Args:
        event_seqs (list of list of 2-tuples):
        labels: (list of list of some kind of labels (e.g., intensities))
    """

    def __init__(self, event_seqs, labels, label_dtype=torch.float):

        self._event_seqs = [np.asarray(seq) for seq in event_seqs]
        self._labels = [np.asarray(_labels) for _labels in labels]
        self._label_dtype = label_dtype

    def __len__(self):
        return len(self._event_seqs)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self._event_seqs[i]).float(),
            torch.from_numpy(self._labels[i]).to(self._label_dtype),
        )

    @staticmethod
    def collate_fn(batch):
        batch_X, batch_y = zip(*batch)

        return (
            nn.utils.rnn.pad_sequence(batch_X, batch_first=True),
            nn.utils.rnn.pad_sequence(batch_y, batch_first=True),
        )
        

class ExplainableRecurrentPointProcess(nn.Module):
    def __init__(
        self,
        max_mean: float,
        embedding_dim: int = 32,
        hidden_size: int = 32,
        n_bases: int = 4,
        basis_type: str = "normal",
        dropout: float = 0.0,
        rnn: str = "GRU",
        **kwargs,
    ):
        super().__init__()
        self.embed = nn.ParameterDict({})
        self.log_intensities_prior = nn.ParameterDict({})

        self.embedding_dim = embedding_dim
        self.n_bases = n_bases
        self.max_type_index = -1

        self.optim = None

        self.dropout = nn.Dropout(p=dropout)
        self.seq_encoder = getattr(nn, rnn)(
            input_size=embedding_dim + 1,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
        )


        self.bases = [Unity()]
        if basis_type == "equal":
            loc, scale = [], []
            for i in range(n_bases):
                loc.append(i * max_mean / (n_bases - 1))
                scale.append(max_mean / (n_bases - 1))
        elif basis_type == "dyadic":
            L = max_mean / 2 ** (n_bases - 1)
            loc, scale = [0], [L / 3]
            for i in range(1, n_bases):
                loc.append(L)
                scale.append(L / 3)
                L *= 2
        else:
            raise ValueError(f"unrecognized basis_type={basis_type}")

        self.bases.append(Normal(loc=loc, scale=scale))
        self.bases = nn.ModuleList(self.bases)

        self.dropout = nn.Dropout(p=dropout)

        # self.shallow_net = ResidualLayer(hidden_size, n_types * (n_bases + 1))
        self.shallow_net = ResidualLayer(hidden_size, embedding_dim * (n_bases + 1))

        if 'n_types' in kwargs.keys():
            for i in range(kwargs['n_types']):
                self.update_event_type(i, torch.device('cpu'))
        # self.log_intensities_prior = torch.nn.Parameter(torch.randn(self.n_types), requires_grad=True)

    def default_param(self, device):
        # return nn.Parameter(torch.zeros(self.embedding_dim), requires_grad=True).to(device)
        return nn.Parameter(torch.randn(self.embedding_dim), requires_grad=True).to(device)

    def extend_type_num(self, n_types):
        for i in range(n_types):
            if str(i) not in self.embed.keys():
                self.update_event_type(i, self.get_model_device())

    @property
    def current_n_types(self):
        return len(self.embed)

    def update_event_type(self, event_type, device):
        self.embed[str(event_type)] = self.default_param(device)
        # self.log_intensities_prior[str(event_type)] = nn.Parameter(torch.randn(1), requires_grad=True).to(device)
        self.log_intensities_prior[str(event_type)] = nn.Parameter(torch.zeros(1), requires_grad=True).to(device)
        self.max_type_index = max(int(event_type), self.max_type_index)

        # self.optim.param_groups.append({'event-log-prior-%s' % str(event_type): self.log_intensities_prior[str(event_type)]})
        # self.optim.param_groups.append({'event-embed-%s' % str(event_type): self.embed[str(event_type)]})

        # self.optim.param_groups.append({'params': self.log_intensities_prior[str(event_type)]})
        # self.optim.param_groups.append({'params': self.embed[str(event_type)]})

        if self.optim is not None:
            self.optim.add_param_group({'params': self.log_intensities_prior[str(event_type)]})
            self.optim.add_param_group({'params': self.embed[str(event_type)]})

    def setdefault_embed(self, event_type, device=None):
        """
        返回事件类型 event_type的embedding，如果不存在，创建新的embedding
        :param event_type:
        :param device:
        :return:
        """
        if str(event_type) not in self.embed.keys():
            self.update_event_type(event_type, device)

        return self.embed[str(event_type)]

    def return_all_parameters(self,  dim):
        # return torch.stack([self.embed[str(i)] for i in range(self.current_n_types)], dim=dim)
        return torch.stack(
            [self.setdefault_embed(i, self.get_model_device()) for i in range(self.max_type_index+1)],
            dim=dim
        )

    def return_all_log_prior(self):
        # return torch.stack([self.embed[str(i)] for i in range(self.current_n_types)], dim=dim)
        return torch.cat(
            [self.log_intensities_prior[str(i)] for i in range(self.max_type_index+1)]
        )

    def get_model_device(self):
        return next(self.parameters()).device

    def event_type2embedding(self, event_seqs, device=torch.device('cpu')):
        embedding_seqs = []
        for seq in event_seqs:
            embedding_seqs.append(
                torch.stack([
                    torch.cat(
                        [
                            torch.FloatTensor([t]).to(device),
                            self.setdefault_embed(int(event_type), device)
                        ]
                    ) for t, event_type in seq
                ], dim=0)
            )

        embedding_seqs = torch.stack(embedding_seqs, dim=0)
        return embedding_seqs

    def event_prior_intensities(self, event_seqs):
        prior_intensities_seqs = []
        for seq in event_seqs:
            prior_intensities_seqs.append(
                torch.cat([self.log_intensities_prior[str(int(event_type))] for t, event_type in seq])
            )

        embedding_seqs = torch.stack(prior_intensities_seqs, dim=0)
        return embedding_seqs


    # def forward(
    #     self, event_seqs, onehot=False, need_weights=True, target_type=-1, device=None
    # ):
    #
    def forward(
            self, event_seqs, event_type='category', need_weights=True, target_type=-1, device=None
    ):
        """[summary]

        Args:
          event_seqs (Tensor): shape=[batch_size, T, 2]
            or [batch_size, T, 1 + n_types] or [batch_size, T, 1 + embedding_dim].
            The last dimension denotes the timestamp and the type of an event, respectively.

          event_type:
            category: event_seqs (Tensor): shape=[batch_size, T, 2]
            onehot: event_seqs (Tensor): shape=[batch_size, T, 1 + n_types]
            feat: event_seqs (Tensor): shape=[batch_size, T, 1 + embedding_dim]

          need_weights (bool): whether to return the basis weights.

          target_type (int): whether to only predict for a specific type

        Returns:
           log_intensities (Tensor): shape=[batch_size, T, n_types],
             log conditional intensities evaluated at each event for each type
             (i.e. starting at t1).
           weights (Tensor, optional): shape=[batch_size, T, n_types, n_bases],
             basis weights intensities evaluated at each previous event (i.e.,
             tarting at t0). Returned only when `need_weights` is `True`.

        """
        # assert event_seqs.size(-1) == 1 + (
        #     self.n_types if onehot else 1
        # ), event_seqs.size()
        if device is None:
            device = self.get_model_device()

        # if onehot:
        #     event_seqs = self.return_all_parameters(dim=0)
        # else:
        #     event_seqs = self.event_type2embedding(event_seqs, device)

        batch_size, T = event_seqs.size()[:2]

        # (t0=0, t1, t2, ..., t_n)
        ts = F.pad(event_seqs[:, :, 0], (1, 0))
        # (0, t1 - t0, ..., t_{n} - t_{n - 1})
        dt = F.pad(ts[:, 1:] - ts[:, :-1], (1, 0))
        # (0, t1 - t0, ..., t_{n - 1} - t_{n - 2})
        temp_feat = dt[:, :-1].unsqueeze(-1)

        # (0, z_1, ..., z_{n - 1})

        if event_type == 'onehot':
            type_feat = event_seqs[:, :-1, 1:] @ self.return_all_parameters(dim=0)
        elif event_type == 'category':
            type_feat = self.event_type2embedding(event_seqs, device)[:, :-1, 1:]
        elif event_type == 'feat':
            type_feat = event_seqs[:, :-1, 1:]

            # type_feat = event_seqs[:, :-1, 1:]
        # if onehot:
        # else:
        #     type_feat = self.embed(
        #         F.one_hot(event_seqs[:, :-1, 1].long(), self.n_types).float()
        #     )
        type_feat = F.pad(type_feat, (0, 0, 1, 0))

        feat = torch.cat([temp_feat, type_feat], dim=-1)
        history_emb, *_ = self.seq_encoder(feat)
        history_emb = self.dropout(history_emb)

        # [B, T, K or 1, R]
        # log_basis_weights = self.shallow_net(history_emb).view(
        #     batch_size, T, self.n_types, -1
        # )

        log_basis_feat = self.shallow_net(history_emb).view(
            batch_size, T, self.n_bases + 1, self.embedding_dim
        )

        if target_type != -1:
            # log_basis_weights = log_basis_weights[
            #     :, :, target_type : target_type + 1
            # ]
            log_basis_weights = (log_basis_feat * self.setdefault_embed(target_type, device)).sum(-1, keepdim=True)
        else:
            all_type_embed = self.return_all_parameters(dim=1)
            log_basis_weights = log_basis_feat @ all_type_embed

        log_basis_weights = log_basis_weights.transpose(-2, -1).contiguous()

        # [B, T, 1, R]
        basis_values = torch.cat(
            [basis.log_prob(dt[:, 1:, None]) for basis in self.bases], dim=2
        ).unsqueeze(-2)

        log_intensities = (log_basis_weights + basis_values).logsumexp(dim=-1)

        if need_weights:
            return log_intensities, log_basis_weights
        else:
            return log_intensities

    def _eval_cumulants(self, batch, log_basis_weights):
        """Evaluate the cumulants (i.e., integral of CIFs at each location)
        """
        ts = batch[:, :, 0]
        # (t1 - t0, ..., t_n - t_{n - 1})
        dt = (ts - F.pad(ts[:, :-1], (1, 0))).unsqueeze(-1)
        # [B, T, R]
        integrals = torch.cat(
            [
                basis.cdf(dt) - basis.cdf(torch.zeros_like(dt))
                for basis in self.bases
            ],
            dim=-1,
        )
        cumulants = integrals.unsqueeze(2).mul(log_basis_weights.exp()).sum(-1)
        return cumulants

    def _eval_nll(
        self, batch, log_intensities, log_basis_weights, mask, debug=False
    ):

        loss_part1 = (
            -log_intensities.gather(dim=2, index=batch[:, :, 1:].long())
            .squeeze(-1)
            .masked_select(mask)
            .sum()
        )

        loss_part2 = (
            self._eval_cumulants(batch, log_basis_weights)
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
            return (loss_part1 + loss_part2 ) / batch.size(0)

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
        self.optim = optim

        train_metrics = defaultdict(AverageMeter)

        for batch in train_dataloader:

            if device:
                batch = batch.to(device)
            seq_length = (batch.abs().sum(-1) > 0).sum(-1)
            mask = generate_sequence_mask(seq_length)

            log_intensities, log_basis_weights = self.forward(
                batch, need_weights=True, event_type='category'
            )

            nll = self._eval_nll(
                batch, log_intensities, log_basis_weights, mask
            )

            T = batch[range(batch.shape[0]), seq_length - 1, 0].sum()
            # prior_log_intensities_loss = (
            #      -self.log_intensities_prior.expand(
            #          *(batch.size()[:2]), -1
            #      ).gather(dim=2, index=batch[:, :, 1:].long())
            #      .squeeze(-1)
            #      .masked_select(mask)
            #      .sum()
            #  ) + self.log_intensities_prior.exp().sum() * T

            prior_log_intensities_loss = (
                                             -self.return_all_log_prior().expand(
                                                 *(batch.size()[:2]), -1
                                             ).gather(dim=2, index=batch[:, :, 1:].long())
                                             .squeeze(-1)
                                             .masked_select(mask)
                                             .sum()
                                         ) + self.return_all_log_prior().exp().sum() * T

            baselines = F.pad(batch[:, :, :1], (0, self.current_n_types))
            baselines_log_intensities = self.forward(
                baselines,
                event_type='onehot',
                need_weights=False
            )
            baselines_loss = ((baselines_log_intensities - self.return_all_log_prior().detach())**2).sum()
            if kwargs["l2_reg"] > 0:
                l2_reg = (
                    kwargs["l2_reg"]
                    * log_basis_weights.permute(2, 3, 0, 1)
                    .masked_select(mask)
                    .exp()
                    .pow(2)
                    .mean()
                )
            else:
                l2_reg = 0.0
            loss = nll + l2_reg + prior_log_intensities_loss + baselines_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # train_metrics["loss"].update(loss, batch.size(0))
            # train_metrics["nll"].update(nll, batch.size(0))
            # train_metrics["l2_reg"].update(l2_reg, seq_length.sum())
            # train_metrics["acc"].update(
            #     self._eval_acc(batch, log_intensities, mask), seq_length.sum()
            # )

            def tofloat(x):
                return float(x.detach().cpu())

            train_metrics["loss"].update(tofloat(loss), batch.size(0))
            train_metrics["nll"].update(tofloat(nll), batch.size(0))
            train_metrics["l2_reg"].update(l2_reg, seq_length.sum())
            train_metrics["acc"].update(
                tofloat(self._eval_acc(batch, log_intensities, mask)), seq_length.sum()
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

                log_intensities, log_basis_weights = self.forward(
                    batch, need_weights=True, event_type='category'
                )
                nll = self._eval_nll(
                    batch, log_intensities, log_basis_weights, mask
                )

                metrics["nll"].update(nll, batch.size(0))
                metrics["acc"].update(
                    self._eval_acc(batch, log_intensities, mask),
                    seq_length.sum(),
                )

        return metrics

    def predict_next_event(
        self, dataloader, predict_type=False, n_samples=100, device=None
    ):
        """[summary]

        Args:
            dataloader (DataLoader):
            predict_type (bool, optional): Defaults to False.
            device (optional): Defaults to None.

        Raises:
            NotImplementedError: if `predict_type = True`.

        Returns:
            event_seqs_pred (List[List[Union[Tuple, float]]]):
        """

        basis_max_vals = torch.cat([basis.maximum for basis in self.bases]).to(
            device
        )

        event_seqs_pred = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)

                seq_length = (batch.abs().sum(-1) > 0).sum(-1)
                mask = generate_sequence_mask(seq_length)
                N = seq_length.sum()

                _, log_basis_weights = self.forward(batch, need_weights=True)
                # sum up weights for all event types
                basis_weights = log_basis_weights.exp().sum(dim=2)
                # [N, R]
                basis_weights = basis_weights.masked_select(
                    mask.unsqueeze(-1)
                ).view(N, -1)

                t = torch.zeros(N * n_samples, device=device)
                sampled_events_type = -1 * torch.ones(
                    (N * n_samples, self.current_n_types), dtype=torch.int, device=device
                )
                # the index for unfinished samples
                idx = torch.arange(N * n_samples, device=device)
                M = basis_weights[idx // n_samples] @ basis_max_vals
                while len(idx) > 0:
                    # get the index for the corresponding basis_weights

                    # region 操作看不懂？？？
                    # todo
                    idx1 = idx // n_samples
                    # M_idx = M[idx1]
                    M_idx = M[idx]
                    # endregion

                    dt = torch.distributions.Exponential(rate=M_idx).sample()
                    t[idx] += dt
                    U = torch.rand(len(idx), device=device)

                    basis_values = torch.cat(
                        [
                            basis.log_prob(t[idx, None]).exp()
                            for basis in self.bases
                        ],
                        dim=-1,
                    )
                    intensity = (basis_weights[idx1] * basis_values).sum(-1)
                    flag = U < (intensity / M_idx)
                    idx = idx[~flag]

                t_pred = t.view(-1, n_samples).mean(-1)
                i = 0
                for b, L in enumerate(seq_length):
                    # reconstruct the actually timestamps
                    seq = t_pred[i : i + L] + F.pad(
                        batch[b, : L - 1, 0], (1, 0)
                    )

                    basis_values = torch.cat(
                        [
                            basis.log_prob(t_pred[i: i + L, None]).exp()
                            for basis in self.bases
                        ],
                        dim=-1,
                    )
                    # t_pred[i: i + L]
                    events_type_intensity = basis_weights[i: i + L] * basis_values
                    events_type_max_intensity = events_type_intensity.argmax(dim=-1)

                    # TODO: pad the event type as type prediction hasn't been
                    # implemented yet.
                    # seq = F.pad(seq[:, None], (0, 1)).cpu().numpy()

                    # 实现序列类型采样
                    seq = torch.cat([seq, events_type_max_intensity[:, None]], dim=-1)

                    event_seqs_pred.append(seq)
                    i += L

        return event_seqs_pred

    def get_infectivity(
        self,
        dataloader,
        device=None,
        steps=50,
        occurred_type_only=False,
        **kwargs,
    ):
        def func(X, target_type):
            _, log_basis_weights = self.forward(
                X, event_type='onehot', need_weights=True, target_type=target_type
            )
            cumulants = self._eval_cumulants(X, log_basis_weights)
            # drop index=0 as it corresponds to (t_0, t_1)
            return cumulants[:, 1:]

        set_eval_mode(self)
        # freeze the model parameters to reduce unnecessary backpropogation.
        for param in self.parameters():
            param.requires_grad_(False)

        A = torch.zeros(self.current_n_types, self.current_n_types, device=device)
        type_counts = torch.zeros(self.current_n_types, device=device).long()

        for batch in tqdm(dataloader):
            if device:
                batch = batch.to(device)

            batch_size, T = batch.size()[:2]
            seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)

            inputs = torch.cat(
                [
                    batch[:, :, :1],
                    F.one_hot(batch[:, :, 1].long(), self.current_n_types).float(),
                ],
                dim=-1,
            )
            baselines = F.pad(inputs[:, :, :1], (0, self.current_n_types))
            mask = generate_sequence_mask(seq_lengths - 1, device=device)

            if occurred_type_only:
                occurred_types = set(
                    batch[:, :, 1]
                    .masked_select(generate_sequence_mask(seq_lengths))
                    .long()
                    .tolist()
                )
            else:
                occurred_types = range(self.current_n_types)

            event_scores = torch.zeros(
                self.current_n_types, batch_size, T - 1, device=device
            )
            for k in occurred_types:
                ig = batch_integrated_gradient(
                    partial(func, target_type=k),
                    inputs,
                    baselines=baselines,
                    mask=mask.unsqueeze(-1),
                    steps=steps,
                )
                event_scores[k] = ig[:, :-1].sum(-1)

            # shape=[K, B, T - 1]
            A.scatter_add_(
                1,
                index=batch[:, :-1, 1]
                .long()
                .view(1, -1)
                .expand(self.current_n_types, -1),
                src=event_scores.view(self.current_n_types, -1),
            )

            ks = (
                batch[:, :, 1]
                .long()
                .masked_select(generate_sequence_mask(seq_lengths))
            )
            type_counts.scatter_add_(0, index=ks, src=torch.ones_like(ks))

        # plus one to avoid divison by zero
        A /= type_counts[None, :].float() + 1

        return A.detach().cpu()

    def get_seq_contribution(
            self,
            batch,
            device=None,
            steps=50,
            occurred_type_only=False,
            **kwargs,
    ):
        def func(X, p, batch):
            log_intensities = self.forward(
                X, event_type='feat', need_weights=False
            )
            # return log_intensities
            log_intensities_specific = torch.gather(
                log_intensities[:, p],
                dim=-1,
                index=batch[:, p, 1:].repeat(X.size(0), 1).long()
            )
            return log_intensities_specific

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

        inputs = self.event_type2embedding(batch)
        # inputs = torch.cat(
        #     [
        #         batch[:, :, :1],
        #         F.one_hot(batch[:, :, 1].long(), self.current_n_types).float(),
        #     ],
        #     dim=-1,
        # )
        baselines = F.pad(inputs[:, :, :1], (0, self.embedding_dim))

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
            mask_p = (seq_lengths > pos)
            ig = batch_integrated_gradient(
                partial(func, p=pos, batch=batch),
                inputs,
                baselines=baselines,
                mask=mask_p,
                steps=steps,
                # idx=pos
            )
            event_scores[:, pos] = ig[:, :-1].sum(-1)

        log_intensities_events = self.forward(inputs, event_type='feat', need_weights=False).gather(dim=2, index=batch[:, :, 1:].long()).squeeze(dim=-1)
        base_log_intensities_events = self.forward(baselines, event_type='feat', need_weights=False).gather(dim=2, index=batch[:, :, 1:].long()).squeeze(dim=-1)

        prior_log_intensities_events = self.event_prior_intensities(batch)
        # prior_log_intensities_events = self.log_intensities_prior.expand(*(batch.size()[:2]), -1).gather(dim=2, index=batch[:, :, 1:].long()).squeeze(dim=-1)

        return event_scores.detach().cpu(), log_intensities_events.detach().cpu(), base_log_intensities_events.detach().cpu(), prior_log_intensities_events.detach().cpu()

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


