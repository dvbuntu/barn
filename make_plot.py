import seaborn as sns
import matplotlib.pyplot as plt
#plt.ion()
import pandas as pd

dfile = 'out/res_all.csv'

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
                col_wrap=4,
                split=True, inner='stick')
#g.set_xticklabels(rotation=20)
plt.tight_layout(w_pad=1)
plt.savefig('pres_violin_results.png')
plt.close()
# regular bar plot
g = sns.catplot(x="Model", y="RMSE",
                 col="Dataset",
                data=df2, kind="bar",
                height=6, aspect=.7,
                sharey=False, legend_out=False,
                col_wrap=4
               )
#g.set_xticklabels(rotation=20)
plt.tight_layout(w_pad=1)
plt.savefig('pres_results.png')

# print table
M = df.groupby('Dataset').mean()
S = df.groupby('Dataset').std()
print(M,S)
