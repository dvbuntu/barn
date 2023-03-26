import seaborn as sns
import matplotlib.pyplot as plt
#plt.ion()
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f','--dfile', default='results/res_all.csv', help='Results output file to read')
parser.add_argument('-o','--ofile', default='results/pres_', help='Plots prepend name')
parser.add_argument('-c','--cname', default='RMSE', help='Variable name to plot')
parser.add_argument('-l','--log', default=False, action='store_true', help='Plot on log scale')

args = parser.parse_args()
dfile = args.dfile
ofile = args.ofile
cname = args.cname
log = args.log

df = pd.read_csv(dfile)

models = df.columns[1:]
rows = []
for i,r in df.iterrows():
    dataname = r['Dataset'].split(',')
    for m in models:
        rows.append([dataname[0], m, r[m]])

df2 = pd.DataFrame(rows, columns=['Dataset', 'Model', cname])

# violin plot
g = sns.catplot(x="Model", y=cname,
                col="Dataset",
                data=df2, kind="violin",
                height=6, aspect=.7,
                sharey=False, legend_out=False,
                col_wrap=5,
                split=True, inner='stick')
#g.set_xticklabels(rotation=20)
if log:
    plt.yscale('log')
plt.tight_layout(w_pad=1)
plt.savefig(ofile + 'violin_results.png')
plt.close()
# regular bar plot
g = sns.catplot(x="Model", y=cname,
                 col="Dataset",
                data=df2, kind="bar",
                height=6, aspect=.7,
                sharey=False, legend_out=False,
                col_wrap=5
               )
#g.set_xticklabels(rotation=20)
if log:
    plt.yscale('log')
plt.tight_layout(w_pad=1)
plt.savefig(ofile + 'results.png')
plt.close()


# print table
M = df.groupby('Dataset').mean()
S = df.groupby('Dataset').std()
print(M)
print(S)

# normalized boxplot
mins = df2.groupby('Dataset').min()
mins[f'min {cname}'] = mins[cname]
df2 = df2.join(mins[f'min {cname}'], on='Dataset')
df2[f'Rel {cname}'] = df2[cname] / df2[f'min {cname}']

g = sns.boxplot(x='Model', y=f'Rel {cname}', data=df2)
if log:
    plt.yscale('log')
plt.title(f'Relative {cname} Across All Data sets')
if cname == 'RMSE':
    plt.ylim(1,10) # beyond 10x worse isn't worth reporting
g.grid(axis='y', which='both')
plt.tight_layout(w_pad=1)
plt.savefig(ofile + 'box_results.png')
plt.close()

