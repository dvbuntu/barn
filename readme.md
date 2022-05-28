# Bayesian Additive Regression Networks

Replacing decision trees of BART with small neural networks.  Prototype research code. 

* [Draft Research Paper](https://drive.google.com/file/d/1ErKl5fcJgivNsXT5Xmlw1uoSBYWtrGhk/view?usp=sharing)
* [Presentation](https://drive.google.com/file/d/1L4uKy3YZkIoTtBtIy6AgyQdLrYhKbHqx/view?usp=sharing)

## Requirements

* `sklearn`
* `numpy`
* `scipy`
* `rpy2`
* `BART` (R package)

## Data

Datasets are available on [drive](https://drive.google.com/drive/folders/1rgpF1TqEEEjEI5xPpv06baCAhjc16fPv?usp=sharing), but they're just lightly processed versions of standard benchmarks.

## Usage

```
usage: simple_nn_mcmc.py [-h] [-l MEAN_NEURONS] [-n NUM_NETS] [-b BURN] [-i ITER]
                         [--nrun NRUN] [-o OUT_PREPEND] [-s DATASETS] [--lr LR]
                         [--no-warn] [--bart] [-m NUM_BATCH]

Bayesian Additive Regression Networks

options:
  -h, --help            show this help message and exit
  -l MEAN_NEURONS, --mean_neurons MEAN_NEURONS
                        Mean number of neurons
  -n NUM_NETS, --num_nets NUM_NETS
                        Number of networks
  -b BURN, --burn BURN  Number of burn-in iterations
  -i ITER, --iter ITER  Number of post-burn-in iterations
  --nrun NRUN           Number of independent runs to try
  -o OUT_PREPEND, --out-prepend OUT_PREPEND
                        prepend to all output (with say, path)
  -s DATASETS, --datasets DATASETS
                        Add this data (wisconsin, etc)
  --lr LR               Learning rate
  --no-warn
  --bart
  -m NUM_BATCH, --num_batch NUM_BATCH
                        Number of batches for batch mean analysis
```

## Examples

Run on a single dataset with defaults:

`python3 simple_nn_mcmc.py -s mpg`

Multiple datasets with no warnings and 16 networks in each ensemble:

`python3 simple_nn_mcmc.py -s mpg -s wisconsin -n 16`

See `dataloader.py` for dataset options, currently:

* boston
* concrete
* crimes
* fires
* mpg
* protein
* wisconsin
* random (generates random regression problem)

Output defaults to current directory.  Trained model(s) stored in `[dataset]_BARN_ensemble.p` pickle files.  Results in `npy` files.  Error rate barplots and batch means analysis also stored in images if requested.  Overall error table saved in `res_all.csv` (for reading by `make_plots.py`).

# TODO

* Automatically download data if directory not found (already implemented for some datasets like boston).
* Double check residual calculations.
* Run on more datasets.
* Speed/scale
    * Port NN training to TensorFlow/Keras and run on GPU for speed.
    * Embed ensemble in a single large NN and only mask off relevant portions for speed.
    * Scale up to 100 networks.
* Error analysis
    * Prove/disprove fast convergence of BARN MCMC process.
    * Compute thorough empirical error analysis (integral autocorrelation time using `acor`).
* Better theoretical justification for priors and transition probabilities.
