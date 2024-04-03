import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_fig(datafile):
    df = pd.read_csv(datafile, header=None, names=['context', 'confidence', 'one-token', 'cot'])
    df_1t = df.groupby('confidence')['one-token'].mean()
    df_cot = df.groupby('confidence')['cot'].mean()

    figs, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax in axs:
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Stepback Rate')
        ax.set_ylim([0, 1])
    
    sample_size = len(df.index)/len(df_1t.index)
    axs[0].errorbar(df_1t.index, df_1t, yerr=np.sqrt(df_1t*(1-df_1t)/sample_size), fmt='o', linestyle='', capsize=5, c='k')
    axs[0].set_title('One-Token')
    axs[1].errorbar(df_cot.index, df_cot, yerr=np.sqrt(df_cot*(1-df_cot)/sample_size), fmt='o', linestyle='', capsize=5, c='k')
    axs[1].set_title('COT')
    plt.savefig(datafile.replace('.csv', '.png'))

    def confusion_matrix(col1, col2):
        z = list(zip(col1, col2))
        a = sum([1 for (x,y) in z if x and y])
        b = sum([1 for (x,y) in z if x and not y])
        c = sum([1 for (x,y) in z if not x and y])
        d = sum([1 for (x,y) in z if not x and not y])
        print(f"Both: {a}")
        print(f"Col1: {b}")
        print(f"Col2: {c}")
        print(f"Neither: {d}")
        return a,b,c,d

    print(f"\n{datafile}:")
    a,b,c,d = confusion_matrix(df['one-token'], df['cot'])
    n_acc = (a+d)/len(df.index)
    print(f"One-Token vs COT accuracy: {n_acc*100:.2f} +- {2*np.sqrt(n_acc*(1-n_acc)/len(df.index))*100:.2f}%\n")

make_fig('cpc_validation_results_numerical.csv')
make_fig('cpc_validation_results_verbal.csv')
make_fig('cpc_validation_results_hints.csv')