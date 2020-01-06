"""Examples for real-valued targets."""

import sys
import subprocess
import signal
import argparse
import random
import numpy as np
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from .modules import RunningAverageMeter, ODEJumpFunc
from .utils import poisson_lmbda, exponential_hawkes_lmbda, powerlaw_hawkes_lmbda, self_inhibiting_lmbda, forward_pass, visualize, create_outpath

signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('point_processes')
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--jump_type', type=str, default='none')
parser.add_argument('--paramr', type=str, default='params.pth')
parser.add_argument('--paramw', type=str, default='params.pth')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--nsave', type=int, default=10)
parser.add_argument('--dataset', type=str, default='poisson')
parser.add_argument('--suffix', type=str, default='')

parser.set_defaults(restart=False, evnt_align=False, seed0=False, debug=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--evnt_align', dest='evnt_align', action='store_true')
parser.add_argument('--seed0', dest='seed0', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()

# set up logging to files named with corresponding commit
outpath = create_outpath(args.dataset)
commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
if not args.debug:
    matplotlib.use('agg')
    sys.stdout = open(outpath + '/' + commit + '.log', 'w')
    sys.stderr = open(outpath + '/' + commit + '.err', 'w')


def read_timeseries(filename, scale=1.0, feature_scale=1.0, num_seqs=sys.maxsize):
    """Read a file containing timeseries of events. Each row corresponds to one timeseries, and ends with ';'.

    Outputs a list of lists of tuples. [[(ts[i] * scale, (ts[i] - ts[i-1]) * scale * feature_scale), ...], ...]
    """
    with open(filename) as f:
        seqs = f.readlines()[:num_seqs]

    timeseries = []
    for seq in seqs:
        ts = seq.split(';')[0].split()
        events = [] if len(ts) == 0 else [(float(ts[0])*scale, [float(ts[0])*scale*feature_scale])]
        for i in range(1, len(ts)):
            events.append((float(ts[i])*scale, [(float(ts[i]) - float(ts[i-1]))*scale*feature_scale]))
        timeseries.append(events)

    return timeseries


if __name__ == '__main__':
    # write all parameters to output file
    print(args, flush=True)

    # fix seeding for randomness
    if args.seed0:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    dim_c, dim_h, dim_N, dt, tspan = 5, 5, 1, 0.05, (0.0, 100.0)

    # scale the feature
    feature_scale = 0.1

    path = "./data/point_processes/"
    TSTR = read_timeseries(path + args.dataset + "_training.csv", 1.0, feature_scale)
    TSVA = read_timeseries(path + args.dataset + "_validation.csv", 1.0, feature_scale)
    TSTE = read_timeseries(path + args.dataset + "_testing.csv", 1.0, feature_scale)

    # compute the true intensities
    if args.dataset == "poisson":
        lmbda_va_real = poisson_lmbda(tspan[0], tspan[1], dt, 1.0, TSVA)
        lmbda_te_real = poisson_lmbda(tspan[0], tspan[1], dt, 1.0, TSTE)
    elif args.dataset == "exponential_hawkes":
        lmbda_va_real = exponential_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 1.0, TSVA, args.evnt_align)
        lmbda_te_real = exponential_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 1.0, TSTE, args.evnt_align)
    elif args.dataset == "powerlaw_hawkes":
        lmbda_va_real = powerlaw_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 2.0, 1.0, TSVA, args.evnt_align)
        lmbda_te_real = powerlaw_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 2.0, 1.0, TSTE, args.evnt_align)
    elif args.dataset == "self_inhibiting":
        lmbda_va_real = self_inhibiting_lmbda(tspan[0], tspan[1], dt, 0.5, 0.2, TSVA, args.evnt_align)
        lmbda_te_real = self_inhibiting_lmbda(tspan[0], tspan[1], dt, 0.5, 0.2, TSTE, args.evnt_align)

    # initialize / load model
    func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_N,
                       dim_hidden=20, num_hidden=1, ortho=True,
                       jump_type=args.jump_type, evnt_align=args.evnt_align,
                       activation=nn.CELU(), evnt_embedding="continuous")
    c0 = torch.randn(dim_c, requires_grad=True)
    h0 = torch.zeros(dim_h)
    it0 = 0
    optimizer = optim.Adam([{'params': func.parameters()}, {'params': c0, 'lr': 1.0e-2}],
                           lr=1e-4, weight_decay=1e-5)

    if args.restart:
        checkpoint = torch.load(args.paramr)
        func.load_state_dict(checkpoint['func_state_dict'])
        c0 = checkpoint['c0']
        h0 = checkpoint['h0']
        it0 = checkpoint['it0']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss_meter = RunningAverageMeter()

    # if read from history, then fit to maximize likelihood
    it = it0
    if func.jump_type == "read":
        while it < args.niters:
            # clear out gradients for variables
            optimizer.zero_grad()

            # sample a mini-batch, create a grid based on that
            batch_id = np.random.choice(len(TSTR), args.batch_size, replace=False)
            batch = [TSTR[seqid] for seqid in batch_id]

            # forward pass
            tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean, gsvar = forward_pass(func, torch.cat((c0, h0), dim=-1),
                                                                                      tspan, dt, batch, args.evnt_align,
                                                                                      scale=1.0/feature_scale)
            loss_meter.update(loss.item() / len(batch))

            # backward prop
            func.backtrace.clear()
            loss.backward()
            print("iter: {}, current loss: {:10.4f}, running ave loss: {:10.4f}, type error: {}".format(it, loss.item()/len(batch), loss_meter.avg, mete), flush=True)

            # step
            optimizer.step()

            it = it+1

            # validate and visualize
            if it % args.nsave == 0:
                # use the full validation set for forward pass
                tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean, gsvar = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, TSVA, args.evnt_align, scale=1.0/feature_scale)

                # backward prop
                func.backtrace.clear()
                loss.backward()
                print("iter: {}, validation loss: {:10.4f}, type error: {}".format(it, loss.item()/len(TSVA), mete), flush=True)

                # visualize
                tsave_ = torch.tensor([record[0] for record in reversed(func.backtrace)])
                trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
                visualize(outpath, tsave, trace, lmbda, tsave_, trace_, tsave[gtid], lmbda_va_real, tsne, range(len(TSVA)), it, gsmean=gsmean, gsvar=gsvar, scale=1.0/feature_scale)

                # save
                torch.save({'func_state_dict': func.state_dict(), 'c0': c0, 'h0': h0, 'it0': it, 'optimizer_state_dict': optimizer.state_dict()}, outpath + '/' + str(it) + '_' + args.paramw)

    # computing testing error
    tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean, gsvar = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, TSTE, args.evnt_align, scale=1.0/feature_scale)
    visualize(outpath, tsave, trace, lmbda, None, None, tsave[gtid], lmbda_te_real, tsne, range(len(TSTE)), it, appendix="testing", gsmean=gsmean, gsvar=gsvar, scale=1.0/feature_scale)
    print("iter: {}, testing loss: {:10.4f}, type error: {}".format(it, loss.item()/len(TSTE), mete), flush=True)
