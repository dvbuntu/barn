try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
# use sklearn for now, could upgrade to keras later if we want
import sklearn.neural_network as sknn
import sklearn.datasets as skds
from sklearn.utils import check_random_state as crs
import sklearn.model_selection as skms
import sklearn.metrics as metrics
import argparse
import pickle
from dataloader import load_data

parser = argparse.ArgumentParser(description='Bayesian Additive Regression Networks')
parser.add_argument('-l','--mean_neurons', default=4, type=int, help='Mean number of neurons')
parser.add_argument('-n','--num_nets', default=10, type=int, help='Number of networks')
parser.add_argument('-b','--burn', default=100, type=int, help='Number of burn-in iterations')
parser.add_argument('-i','--iter', default=100, type=int, help='Number of post-burn-in iterations')
parser.add_argument('--nrun', default=1, type=int, help='Number of independent runs to try')
parser.add_argument('-o','--out-prepend', default='', type=str, help='prepend to all output (with say, path)')
parser.add_argument('-s','--datasets', action='append',
                        default=[],
                        help='Add this data (wisconsin, etc)')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--no-warn', default=False, action='store_true')
parser.add_argument('--bart', default=False, action='store_true')
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
BIG = 2**32-1
if len(args.datasets) == 0:
    args.datasets = ['random']

class NN(object):
    def __init__(self, num_nodes=10, weight_donor=None, l=10, lr=0.01, r=None):
        self.num_nodes = num_nodes
        # make an NN with a single hidden layer with num_nodes nodes
        self.model = sknn.MLPRegressor([num_nodes], learning_rate_init=lr, random_state=r)
        # l is poisson shape param, expected number of nodes
        self.l = l
        self.lr = lr
        self.r = r
        if weight_donor is not None:
            # inherit the first num_nodes weights from this donor
            donor_num_nodes = weight_donor.num_nodes
            donor_weights = weight_donor.model.coefs_
            donor_intercepts = weight_donor.model.intercepts_
            self.accept_donation(donor_num_nodes, donor_weights, donor_intercepts)

    def save(self, fname):
        params = np.array([self.num_nodes, self.l, self.lr, self.r])
        np.savez_compressed(fname, params=params,
                coefs_=self.model.coefs_,
                intercepts_=self.model.intercepts_)

    def accept_donation(self, donor_num_nodes, donor_weights, donor_intercepts):
        # a big of a workaround to create weight arrays and things
        num_nodes = self.num_nodes
        self.model._random_state = crs(self.r)
        self.model._initialize(np.zeros((1,1),dtype=donor_weights[0].dtype),
                               [donor_weights[0].shape[0], num_nodes, 1],
                               donor_weights[0].dtype)
        if donor_num_nodes == num_nodes:
            self.model.coefs_ = [d.copy() for d in donor_weights]
            self.model.intercepts_ = [d.copy() for d in donor_intercepts]
        elif donor_num_nodes > num_nodes:
            self.model.coefs_ = [donor_weights[0][:,:num_nodes].copy(),
                                 donor_weights[1][:num_nodes].copy()]
            self.model.intercepts_ = [donor_intercepts[0][:num_nodes].copy(),
                                 donor_intercepts[1].copy()]
        else:
            self.model.coefs_[0][:,:donor_num_nodes] = donor_weights[0].copy()
            self.model.coefs_[1][:donor_num_nodes] = donor_weights[1].copy()
            self.model.intercepts_[0][:donor_num_nodes] = donor_intercepts[0].copy()
            self.model.intercepts_[1] = donor_intercepts[1].copy()

    @staticmethod
    def load(fname):
        network = np.load(fname)
        N = NN(network['params'][0],
               l=network['params'][1],
               lr=network['params'][2],
               r=network['params'][3])
        donor_num_nodes = N.num_nodes
        donor_weights = network['coefs_']
        donor_intercepts_ = network['intercepts_']
        self.accept_donation(donor_num_nodes, donor_weights, donor_intercepts)
        return N

    def train(self, X, Y, epochs=10):
        '''Train network from current position with given data'''
        # TODO: figure out how to fix num epochs
        self.model.fit(X,Y)

    def log_prior(self):
        return scipy.stats.poisson.logpmf(self.num_nodes, N.l)

    def log_likelihood(self, X, Y):
        # compute residuals
        yhat = self.model.predict(X)
        resid = Y - yhat
        # compute stddev of these
        std = np.std(resid)
        # normal likelihood
        return np.sum(scipy.stats.norm.logpdf(resid, 0, std))

    def log_acceptance(self, X, Y):
        return self.log_prior+self.log_likelihood(X,Y)

    def log_transition(self, target, q=0.5):
        '''Transition probability from self to target'''
        #return target.log_prior()-self.log_prior()+np.log(q)
        # For now assume simple transition model
        return np.log(q)

    def __repr__(self):
        return f'NN({self.num_nodes}, l={self.l}, lr={self.lr})'

# total acceptable of moving from N to Np given data XY
def A(Np, N, X, Y, q=0.5):
    # disallow empty network...or does this mean kill it off entirely?
    if Np.num_nodes < 1:
        return 0
    num = Np.log_transition(N,q) + Np.log_likelihood(X, Y) + Np.log_prior()
    denom = N.log_transition(Np,1-q) + N.log_likelihood(X, Y) + N.log_prior()
    # assumes only 2 inverse types of transition
    return min(1, np.exp(num-denom))

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

            # load diabetes data
            #X, Y = skds.load_diabetes(return_X_y=True)

        # some initial NNs, all single node
        cyberspace = [NN(1, l=l, lr=lr) for i in range(num_nets)]
        # fit them all initially
        for j,N in enumerate(cyberspace):
            # initialize fit as though all get equal share of Y
            #R = Y - np.sum([N.model.predict(X) for k,N in enumerate(cyberspace) if k != j], axis=0)
            N.train(Xtr,Ytr/num_nets)

        # check initial fit
        Yh = np.sum([N.model.predict(Xte) for N in cyberspace], axis=0)
        r2h = metrics.r2_score(Yte, Yh)
        rmseh = metrics.mean_squared_error(Yte, Yh, squared=False)
        # do optional BART fit
        if args.bart:
            fig, ax = plt.subplots(1,4, sharex=True, sharey=True, squeeze=True)
            fig.set_size_inches(16,4)
        else:
            fig, ax = plt.subplots(1,3, sharex=True, sharey=True, squeeze=True)
            fig.set_size_inches(12,4)
        ax[0].plot([np.min(Yte), np.max(Yte)],[np.min(Yte), np.max(Yte)])
        ax[0].scatter(Yte,Yh, c='orange') # somewhat decent on synth, gets lousy at edge, which makes sense
        ax[0].set_title('Initial BARN')
        ax[0].set_ylabel('Prediction')
        ax[0].text(0.05, 0.85, f'$R^2 = $ {r2h:0.4}\n$RMSE = $ {rmseh:0.4}', transform=ax[0].transAxes)


        # maybe should bias to shrinking to avoid just overfitting?
        # or compute acceptance resid on validation data?
        trans_probs = [0.4, 0.6] # uniform for now
        trans_options = ['grow', 'shrink']

        accepted = 0
        # setup residual array
        S_tr = np.array([N.model.predict(Xtr) for N in cyberspace])
        S_va = np.array([N.model.predict(Xva) for N in cyberspace])
        Rtr = Ytr - (np.sum(S_tr, axis=0) - S_tr[-1])
        Rva = Yva - (np.sum(S_va, axis=0) - S_va[-1])
        phi = np.zeros(total)
        for i in tqdm(range(total)):
            # gibbs sample over the nets
            for j in range(num_nets):
                # compute resid against other nets
                ## Use cached these results, add back most recent and remove current
                ## TODO: double check this is correct
                Rva = Rva - S_va[j-1] + S_va[j]
                Rtr = Rtr - S_tr[j-1] + S_tr[j]
                #check_Rva = Yva - np.sum([N.model.predict(Xva) for k,N in enumerate(cyberspace) if k != j], axis=0)
                #check_Rtr = Ytr - np.sum([N.model.predict(Xtr) for k,N in enumerate(cyberspace) if k != j], axis=0)
                #assert np.allclose(Rva, check_Rva)
                # grab current net in this position
                N = cyberspace[j]
                # create proposed change
                choice = np.random.choice(trans_options, p=trans_probs)
                if choice == 'grow':
                    Np = NN(N.num_nodes+1, weight_donor=N, l=N.l, lr=N.lr, r=np.random.randint(BIG))
                    q = trans_probs[0]
                elif N.num_nodes-1 == 0:
                    continue # don't bother building empty model
                else:
                    Np = NN(N.num_nodes-1, weight_donor=N, l=N.l, lr=N.lr, r=np.random.randint(BIG))
                    q = trans_probs[1]
                Np.train(Xtr,Rtr)
                # determine if we should keep it
                if np.random.random() < A(Np, N, Xva, Rva, q):
                    cyberspace[j] = Np
                    accepted += 1
                    S_tr[j] = Np.model.predict(Xtr)
                    S_va[j] = Np.model.predict(Xva)
            # overall validation error at this MCMC iteration
            phi[i] = np.sqrt(np.mean((Rva - S_va[j])**2))

        # final fit
        Yh2 = np.sum([N.model.predict(Xte) for N in cyberspace], axis=0)
        r2h2 = metrics.r2_score(Yte, Yh2)
        rmseh2 = metrics.mean_squared_error(Yte, Yh2, squared=False)
        ax[1].plot([np.min(Yte), np.max(Yte)],[np.min(Yte), np.max(Yte)])
        ax[1].scatter(Yte,Yh2, c='orange')
        ax[1].set_title('Final BARN')
        ax[1].set_xlabel('Target')
        ax[1].text(0.05, 0.85, f'$R^2 = $ {r2h2:0.4}\n$RMSE = $ {rmseh2:0.4}', transform=ax[1].transAxes)

        # some get big, some are small.  How's this square with the prior?

        # check batch means variance
        mu = np.mean(phi[burn:])
        np.save(f'{args.out_prepend}{dname}_val_resid.npy', phi) # only final saved
        batch_phi = np.mean(phi[burn:].reshape((num_batch, batch_size)), axis=1)
        var = np.sum((batch_phi-mu)**2)/(num_batch*(num_batch-1))
        with open(f'{args.out_prepend}var_all.csv', 'a') as f:
            print(f'{dname}, {var}', file=f)

        # compare to big NN
        tot_neurons = sum([N.num_nodes for N in cyberspace])
        Nb = NN(tot_neurons, lr=0.1)
        Nb.train(Xtr,Ytr)
        Yhb = Nb.model.predict(Xte)
        r2hb = metrics.r2_score(Yte, Yhb)
        rmsehb = metrics.mean_squared_error(Yte, Yhb, squared=False)
        ax[2].plot([np.min(Yte), np.max(Yte)],[np.min(Yte), np.max(Yte)])
        ax[2].scatter(Yte,Yhb, c='orange')
        ax[2].set_title('Equiv Size NN')
        ax[2].text(0.05, 0.85, f'$R^2 = $ {r2hb:0.4}\n$RMSE = $ {rmsehb:0.4}', transform=ax[2].transAxes)
        ## seems just as good here, and a heck of a lot faster
        ## but, how would we have guessed to use this architecture?
        ## and what about harder problems?
        ## diabetes, they're all pretty bad

        # This only saves the last iteration of full models, but that's something
        with open(f'{args.out_prepend}{dname}_BARN_ensemble.p','wb') as f:
            pickle.dump(cyberspace, f)
        with open(f'{args.out_prepend}{dname}_BARN_derived.p','wb') as f:
            pickle.dump(Nb, f)

        if args.bart:
            from bart import fit_bart
            BB, resb = fit_bart([Xtr, Xva, Xte, Ytr, Yva, Yte], ntrees=num_nets)
            Yht = BB.predict(Xte)
            r2ht = metrics.r2_score(Yte, Yht)
            rmseht = resb[2]

            ax[3].plot([np.min(Yte), np.max(Yte)],[np.min(Yte), np.max(Yte)])
            ax[3].scatter(Yte,Yht, c='orange')
            ax[3].set_title('BART')
            ax[3].text(0.05, 0.85, f'$R^2 = $ {r2ht:0.4}\n$RMSE = $ {rmseht:0.4}', transform=ax[3].transAxes)

            with open(f'{args.out_prepend}{dname}_BART.p','wb') as f:
                pickle.dump(BB, f)
            fig.suptitle(f'{dname} BARN, NN, & BART test results')
            fig.savefig(f'{args.out_prepend}{dname}_results.png')
            plt.close()
            with open(f'{args.out_prepend}res_all.csv', 'a') as f:
                print(f'{dname}, {rmseh2}, {rmsehb}, {rmseht}', file=f)
        else:
            fig.suptitle(f'{dname} BARN and NN test results')
            fig.savefig(f'{args.out_prepend}{dname}_results.png')
            plt.close()
            with open(f'{args.out_prepend}res_most.csv', 'a') as f:
                print(f'{dname}, {rmseh2}, {rmsehb}', file=f)

        # Plot phi results
        plt.plot(phi)
        plt.xlabel('MCMC Iteration')
        plt.ylabel('RMSE')
        plt.title(f'{dname} MCMC Error Progression')
        plt.savefig(f'{args.out_prepend}{dname}_phi.png')
        plt.close()
