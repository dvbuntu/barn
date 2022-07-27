try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
# use sklearn for now, could upgrade to keras later if we want
import sklearn.neural_network as sknn
import sklearn.linear_model
import sklearn.datasets as skds
from sklearn.utils import check_random_state as crs
import sklearn.model_selection as skms
import sklearn.metrics as metrics
import argparse
import pickle
import time
import os
from dataloader import load_data
from barn import *

parser = argparse.ArgumentParser(description='Bayesian Additive Regression Networks')
parser.add_argument('-l','--mean_neurons', default=4, type=int, help='Mean number of neurons')
parser.add_argument('-n','--num_nets', default=10, type=int, help='Number of networks')
parser.add_argument('-b','--burn', default=100, type=int, help='Number of burn-in iterations')
parser.add_argument('-i','--iter', default=100, type=int, help='Number of post-burn-in iterations')
parser.add_argument('--nrun', default=1, type=int, help='Number of independent runs to try')
parser.add_argument('-o','--out-prepend', default='results/', type=str, help='prepend to all output (with say, path)')
parser.add_argument('-s','--datasets', action='append',
                        default=[],
                        help='Add this data (wisconsin, etc)')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--no-warn', default=False, action='store_true')
parser.add_argument('--bart', default=False, action='store_true')
parser.add_argument('--big-nn', default=False, action='store_true')
parser.add_argument('--ols', default=False, action='store_true')
parser.add_argument('-m','--num_batch', default=20, type=int, help='Number of batches for batch mean analysis')

args = parser.parse_args()

if args.no_warn:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

l = args.mean_neurons
num_nets = args.num_nets
lr = args.lr
burn = args.burn
total = burn + args.iter
num_batch = args.num_batch
batch_size = args.iter//num_batch
BIG = 2**31-1 # biggest signed 4byte int
if len(args.datasets) == 0:
    args.datasets = ['random']

# if new results files, prepend header
models = ['BARN']
if args.big_nn:
    models.append('Big NN')
if args.bart:
    models.append('BART')
if args.ols:
    models.append('OLS')

dirname = os.path.dirname(args.out_prepend)
os.makedirs(dirname, exist_ok=True)

if not os.path.isfile(f'{args.out_prepend}res_all.csv'):
    with open(f'{args.out_prepend}res_all.csv', 'a') as f:
        print(', '.join(['Dataset'] + models), file=f)
if not os.path.isfile(f'{args.out_prepend}time_all.csv'):
    with open(f'{args.out_prepend}time_all.csv', 'a') as f:
        print(', '.join(['Dataset'] + models), file=f)

for run_num in range(args.nrun):
    for dname in args.datasets:
        print(dname)
        if dname == 'random':
            # synth data
            sig = 10
            X, Y = skds.make_regression(n_samples=1000, n_features=10, n_informative=8, noise=sig)

            # split into train, valid, test
            Xtr, XX, Ytr, YY = skms.train_test_split(X,Y, test_size=0.5) # training
            Xva, Xte, Yva, Yte = skms.train_test_split(XX,YY, test_size=0.5) # valid and test
        else:
            data = load_data(dname, seed=54+run_num)
            Xtr, Xva, Xte, Ytr, Yva, Yte = data
            # fix shape of Y
            Ytr = Ytr.reshape(-1)
            Yva = Yva.reshape(-1)
            Yte = Yte.reshape(-1)

        # Fill the BARN with some hay
        cur_time = time.time()
        nets = BARN(num_nets, dname=dname)
        nets.setup_nets(l=l, lr=lr)
        nets.train(Xtr, Ytr, Xva, Yva, Xte, Yte, total)
        time_list = [dname, str(time.time()-cur_time)]

        # check batch means variance
        batch_var = nets.batch_means(num_batch, batch_size, np_out=f'{args.out_prepend}{dname}_val_resid.npy', outfile=f'{args.out_prepend}var_all.csv', burn=burn)

        # Setup basic viz
        fig_phi = nets.phi_viz(f'{args.out_prepend}{dname}_phi.png')
        initial=False
        fig, ax, rmseh2 = nets.viz(f'{args.out_prepend}{dname}_results.png',
                       extra_slots=args.big_nn+args.bart+args.ols,
                       close=False,
                       initial=initial)

        # This only saves the last iteration of full models, but that's something
        nets.save(f'{args.out_prepend}{dname}_BARN_ensemble.p')

        out_list = [dname, str(rmseh2)]

        if args.big_nn:
            # compare to big NN
            tot_neurons = sum([N.num_nodes for N in nets.cyberspace])
            cur_time = time.time()
            Nb = NN(tot_neurons, lr=0.1)
            Nb.train(Xtr,Ytr)
            time_list.append(str(time.time()-cur_time))
            Yhb = Nb.model.predict(Xte)
            r2hb = np.abs(metrics.r2_score(Yte, Yhb))
            rmsehb = metrics.mean_squared_error(Yte, Yhb, squared=False)
            nn_time = time.time()
            out_list.append(str(rmsehb))
            ax[1+initial].plot([np.min(Yte), np.max(Yte)],[np.min(Yte), np.max(Yte)])
            ax[1+initial].scatter(Yte,Yhb, c='orange')
            ax[1+initial].set_title('Equiv Size NN')
            ax[1+initial].text(0.05, 0.85, f'$R^2 = $ {r2hb:0.4}\n$RMSE = $ {rmsehb:0.4}', transform=ax[1+initial].transAxes)
            with open(f'{args.out_prepend}{dname}_BARN_derived.p','wb') as f:
                pickle.dump(Nb, f)


        if args.bart:
            from bart import fit_bart
            cur_time = time.time()
            BB, resb = fit_bart([Xtr, Xva, Xte, Ytr, Yva, Yte], ntrees=num_nets)
            time_list.append(str(time.time()-cur_time))
            Yht = BB.predict(Xte)
            r2ht = metrics.r2_score(Yte, Yht)
            rmseht = resb[2]
            out_list.append(str(rmseht))

            ax[1+initial+args.big_nn].plot([np.min(Yte), np.max(Yte)],[np.min(Yte), np.max(Yte)])
            ax[1+initial+args.big_nn].scatter(Yte,Yht, c='orange')
            ax[1+initial+args.big_nn].set_title('BART')
            ax[1+initial+args.big_nn].text(0.05, 0.85, f'$R^2 = $ {r2ht:0.4}\n$RMSE = $ {rmseht:0.4}', transform=ax[1+args.big_nn].transAxes)

            with open(f'{args.out_prepend}{dname}_BART.p','wb') as f:
                pickle.dump(BB, f)

        if args.ols:
            cur_time = time.time()
            ols_model = sklearn.linear_model.LinearRegression()
            ols_model.fit(Xtr, Ytr)
            time_list.append(str(time.time()-cur_time))
            Yho = ols_model.predict(Xte)
            r2ho = metrics.r2_score(Yte, Yho)
            rmseho = metrics.mean_squared_error(Yte, Yho, squared=False)
            out_list.append(str(rmseho))
            ax[1+initial+args.big_nn+args.bart].plot([np.min(Yte), np.max(Yte)],[np.min(Yte), np.max(Yte)])
            ax[1+initial+args.big_nn+args.bart].scatter(Yte,Yho, c='orange')
            ax[1+initial+args.big_nn+args.bart].set_title('OLS')
            ax[1+initial+args.big_nn+args.bart].text(0.05, 0.85, f'$R^2 = $ {r2ho:0.4}\n$RMSE = $ {rmseho:0.4}', transform=ax[1+initial+args.big_nn+args.bart].transAxes)
            with open(f'{args.out_prepend}{dname}_OLS.p','wb') as f:
                pickle.dump(Nb, f)


        # write the final results
        with open(f'{args.out_prepend}res_all.csv', 'a') as f:
            print(', '.join(out_list), file=f)

        with open(f'{args.out_prepend}time_all.csv', 'a') as f:
            print(', '.join(time_list), file=f)

        fig.savefig(f'{args.out_prepend}{dname}_results.png')
        plt.close()
