import argparse
import os
import os.path as osp
import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

dataloader_args = None
device = None
logger = None
args = None
output_path = None

from ..pkg.models.rnn import (
    EventSeqDataset,
    ExplainableRecurrentPointProcess,
)
from ..pkg.models.spnpp import (
SemiParametricPointProcess
)
# from ..pkg.models.rnn_new2 import (
#     EventSeqDataset,
#     ExplainableRecurrentPointProcess,
# )
from ..pkg.models.ode_rnn import ODERecurrentPointProcess
from ..pkg.models.rppn import RecurrentPointProcessNet
from ..pkg.utils.argparser.training import add_subparser_arguments
from ..pkg.utils.evaluation import eval_fns
from ..pkg.utils.logging import get_logger, init_logging
from ..pkg.utils.misc import (
    Timer,
    compare_metric_value,
    export_csv,
    export_json,
    get_freer_gpu,
    makedirs,
    set_rand_seed,
)
from ..pkg.utils.pp import (
    eval_nll_hawkes_exp_kern,
    eval_nll_hawkes_sum_gaussians,
    event_seq_to_counting_proc,
)
from ..pkg.utils.torch import split_dataloader, convert_to_bucketed_dataloader


def get_parser():
    parser = argparse.ArgumentParser(description="Training different models. ")
    subparsers = parser.add_subparsers(
        description="Supported models", dest="model"
    )
    for model in ["ODE-RNN", "SPNPP", "ERPP", "RME", "RPPN", "HExp", "HSG", "NPHC"]:
        add_subparser_arguments(model, subparsers)

    return parser


def get_model(args, n_types):
    if args.model == "ODE-RNN":
        model = ODERecurrentPointProcess(n_types=n_types, **vars(args))
    elif args.model == "SPNPP":
        model = SemiParametricPointProcess(n_types=n_types, **vars(args))
    elif args.model == "ERPP":
        # model = ExplainableRecurrentPointProcess(n_types=n_types, **vars(args))
        model = ExplainableRecurrentPointProcess(n_types=n_types, **vars(args))
    elif args.model == "RPPN":
        model = RecurrentPointProcessNet(n_types=n_types, **vars(args))
    elif args.model == "HExp":
        # 'gd', 'agd', 'bfgs', 'svrg'
        from tick.hawkes import HawkesExpKern
        model = HawkesExpKern(args.decay, solver=args.solver, C=args.penalty, verbose=args.verbose)
    elif args.model == "HSG":
        from tick.hawkes import HawkesSumGaussians

        model = HawkesSumGaussians(
            args.max_mean,
            n_gaussians=args.n_gaussians,
            C=args.penalty,
            n_threads=args.n_threads,
            verbose=args.verbose,
        )
    elif args.model == "NPHC":
        from tick.hawkes import HawkesCumulantMatching

        model = HawkesCumulantMatching(
            integration_support=args.integration_support,
            C=args.penalty,
            verbose=args.verbose,
        )
    else:
        raise ValueError(f"Unsupported model={args.model}")

    return model


def get_device(cuda, dynamic=False):
    if torch.cuda.is_available() and args.cuda:
        if dynamic:
            device = torch.device("cuda", get_freer_gpu(by="n_proc"))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def get_hparam_str(args):
    if args.model == "ERPP" or args.model == "SPNPP":
        hparams = ["max_mean", "n_bases", "hidden_size", "lr"]
    else:
        hparams = []

    return ",".join("{}={}".format(p, getattr(args, p)) for p in hparams)


def train_nn_models(model, event_seqs, args):

    train_dataloader = DataLoader(
        EventSeqDataset(event_seqs), **dataloader_args
    )

    train_dataloader, valid_dataloader = split_dataloader(
        train_dataloader, 8 / 9
    )
    if "bucket_seqs" in args and args.bucket_seqs:
        train_dataloader = convert_to_bucketed_dataloader(
            train_dataloader, key_fn=len
        )
    valid_dataloader = convert_to_bucketed_dataloader(
        valid_dataloader, key_fn=len, shuffle_same_key=False
    )

    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(), lr=args.lr
    )

    model.train()
    best_metric = float("nan")
    best_epoch = -1

    for epoch in range(args.epochs):
        train_metrics, valid_metrics = model.train_epoch(
            train_dataloader,
            optimizer,
            valid_dataloader,
            device=device,
            **vars(args),
        )

        msg = f"[Training] Epoch={epoch}"
        for k, v in train_metrics.items():
            msg += f", {k}={v.avg:.4f}"
        logger.info(msg)
        msg = f"[Validation] Epoch={epoch}"
        for k, v in valid_metrics.items():
            msg += f", {k}={v.avg:.4f}"
        logger.info(msg)

        if compare_metric_value(
            valid_metrics[args.tune_metric].avg, best_metric, args.tune_metric
        ):
            if epoch > args.epochs // 2:
                logger.info(f"Found a better model at epoch {epoch}.")
            best_metric = valid_metrics[args.tune_metric].avg
            torch.save(model.state_dict(), osp.join(output_path, "model.pt"))
            best_epoch = epoch
        elif epoch > args.epochs // 2 and epoch - best_epoch > 40:

            logger.info(
                'The NLL does not decrease in last %d epochs. Early stopping at epoch=%d' % (epoch-best_epoch, epoch)
            )
            break

    model.load_state_dict(torch.load(osp.join(output_path, "model.pt")))

    return model


def eval_nll(model, event_seqs, args):
    if args.model in ["ODERNN", "SPNPP", "RME", "ERPP", "RPPN"]:

        dataloader = DataLoader(
            EventSeqDataset(event_seqs), shuffle=False, **dataloader_args
        )

        metrics = model.evaluate(dataloader, device=device)
        logger.info(
            "[Test]"
            + ", ".join(f"{k}={v.avg:.4f}" for k, v in metrics.items())
        )
        nll = metrics["nll"].avg.item()

    elif args.model == "HSG":
        nll = eval_nll_hawkes_sum_gaussians(event_seqs, model, verbose=True)

    elif args.model == "HExp":
        nll = eval_nll_hawkes_exp_kern(event_seqs, model, verbose=True)
    else:
        nll = float("nan")
        print("not supported yet")

    return nll


def predict_next_event(model, event_seqs, args):
    if args.model in ["ODERNN", "SPNPP", "ERPP", "RPPN"]:
        dataloader = DataLoader(
            EventSeqDataset(event_seqs), shuffle=False, **dataloader_args
        )
        event_seqs_pred = model.predict_next_event(dataloader, device=device)
    elif args.model == "HExp":
        from ..pkg.utils.pp import predict_next_event_hawkes_exp_kern

        event_seqs_pred = predict_next_event_hawkes_exp_kern(
            event_seqs, model, verbose=True
        )
    else:
        print(
            "Predicting next event is not supported for "
            f"model={args.model} yet."
        )
        event_seqs_pred = None

    return event_seqs_pred


def get_infectivity_matrix(model, event_seqs, args):

    if args.model in ["RME", "ERPP", "RPPN"]:
        _dataloader_args = dataloader_args.copy()
        if "attr_batch_size" in args and args.attr_batch_size:
            _dataloader_args.update(batch_size=args.attr_batch_size)

        dataloader = DataLoader(
            EventSeqDataset(event_seqs), **_dataloader_args
        )
        dataloader = convert_to_bucketed_dataloader(dataloader, key_fn=len)
        infectivity = model.get_infectivity(dataloader, device, **vars(args))
    else:
        infectivity = model.get_kernel_norms()

    return infectivity
