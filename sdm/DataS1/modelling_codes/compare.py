import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

plt.ion()

model_files = glob.glob('summary_results/*.csv')

fig, ax = plt.subplots(2,2, sharex=True, sharey=False, squeeze=True)

for m in model_files:
    name = m[21:-4]
    res = pd.read_csv(m)
    ax[0,0].plot(res['roc'])
    ax[0,1].plot(res['prg'])
    ax[1,0].plot(res['cor'])
    ax[1,1].plot(res['time'], label=name)

ax[0,0].set_title('ROC')
ax[0,1].set_title('PRG')
ax[1,0].set_title('COR')
ax[1,1].set_title('Time (s)')

ax[0,0].set_title('ROC')
ax[0,1].set_title('PRG')
ax[1,0].set_title('COR')
ax[1,1].set_title('Time (s)')

ax[1,1].legend()

