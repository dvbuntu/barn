import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.tri as tri

#plt.ion()
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f','--dfile', default='results/time_all.csv', help='Results output file to read')
parser.add_argument('-o','--ofile', default='results/time_', help='Plots prepend name')

args = parser.parse_args()
dfile = args.dfile
ofile = args.ofile

df = pd.read_csv(dfile)

models = df.columns[1:]
rows = []
for i,r in df.iterrows():
    dataname = r['Dataset'].split(',')
    for m in models:
        rows.append([dataname[0], m, r[m]])

df2 = pd.DataFrame(rows, columns=['Dataset', 'Model', 'Time'])


ndata = {'boston':506,
        'concrete':1030,
        'crimes':1994,
        'diabetes':442,
        'fires':517,
        'isotope':700,
        'mpg':398,
        'random':1000,
        'wisconsin':194}

nfeature = {'boston':13,
        'concrete':9,
        'crimes':101,
        'diabetes':10,
        'fires':10,
        'isotope':1,
        'mpg':7,
        'random':10,
        'wisconsin':32}

for k in ndata.keys():
    df2.loc[df2.Dataset == k, 'N Features'] = nfeature[k]
    df2.loc[df2.Dataset == k, 'N Data'] = ndata[k]

df2 = df2.astype({'N Features':'int32','N Data':'int32'})

bdf = df2[df2.Model==' BARN']
bdf['Data x Features'] = bdf['N Features'] * bdf['N Data']

fig, ax = plt.subplots(1,3, sharey=True, sharex=False, figsize=(12,4))

sns.scatterplot(data=bdf, x='N Data', y='Time', ax=ax[0], alpha=0.5)
ax[0].set_xscale('log')
ax[0].set_xlim(bdf['N Data'].min()*.9, bdf['N Data'].max()*1.1)
xmin, xmax = ax[0].get_xlim()
labs = np.logspace(np.log10(xmin), np.log10(xmax), num=5, dtype=int)
ax[0].set_xticks([], minor=True)
ax[0].set_xticks(labs, labs)
ax[0].set_ylabel('Time (s)')
ax[0].set_xlabel('N Data (log scale)')

sns.scatterplot(data=bdf, x='N Features', y='Time', ax=ax[1], alpha=0.5)
ax[1].set_xscale('log')
ax[1].set_ylabel('Time (s)')
ax[1].set_xlabel('N Features (log scale)')

sns.scatterplot(data=bdf, x='Data x Features', y='Time', ax=ax[2], alpha=0.5)
ax[2].set_xscale('log')
ax[2].set_ylabel('Time (s)')
ax[2].set_xlabel('Data x Features (log scale)')

fig.suptitle('BARN Computation Time Across Problem Sizes')

plt.tight_layout(w_pad=1)
plt.savefig(ofile + 'results.png')

