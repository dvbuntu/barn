import seaborn as sns
import matplotlib.pyplot as plt
#plt.ion()
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f','--dfile', default='results/res_all.csv', help='Results output file to read')
parser.add_argument('-o','--ofile', default='results/pres_', help='Plots prepend name')

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

df2 = pd.DataFrame(rows, columns=['Dataset', 'Model', 'RMSE'])

# violin plot
g = sns.catplot(x="Model", y="RMSE",
                col="Dataset",
                data=df2, kind="violin",
                height=6, aspect=.7,
                sharey=False, legend_out=False,
                col_wrap=5,
                split=True, inner='stick')
#g.set_xticklabels(rotation=20)
plt.tight_layout(w_pad=1)
plt.savefig(ofile + 'violin_results.png')
plt.close()
# regular bar plot
g = sns.catplot(x="Model", y="RMSE",
                 col="Dataset",
                data=df2, kind="bar",
                height=6, aspect=.7,
                sharey=False, legend_out=False,
                col_wrap=5
               )
#g.set_xticklabels(rotation=20)
plt.tight_layout(w_pad=1)
plt.savefig(ofile + 'results.png')

# print table
M = df.groupby('Dataset').mean()
S = df.groupby('Dataset').std()
print(M)
print(S)
