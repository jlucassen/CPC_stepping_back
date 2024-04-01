import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_numerical = pd.read_csv('cpc_validation_results_numerical.csv')
df_verbal = pd.read_csv('cpc_validation_results_verbal.csv')

# figs without grouping by context

df_numerical.columns = ['context', 'confidence', 'one-token', 'cot']
df_verbal.columns = ['context', 'confidence', 'one-token', 'cot']
numerical_onetoken = df_numerical.groupby('confidence')['one-token'].mean()
numerical_cot = df_numerical.groupby('confidence')['cot'].mean()
verbal_onetoken = df_verbal.groupby('confidence')['one-token'].mean()
verbal_cot = df_verbal.groupby('confidence')['cot'].mean()

figs, axs = plt.subplots(2, 2, figsize=(10, 10))
for row in axs: 
    for ax in row:
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Stepback Rate')
        ax.set_ylim([0, 1])

def make_errs(data):
    return np.sqrt(data*(1-data)/len(data))

axs[0,0].errorbar(numerical_onetoken.index, numerical_onetoken, yerr=np.sqrt(numerical_onetoken*(1-numerical_onetoken)/150), fmt='o', linestyle='', capsize=5, c='k')
axs[0,0].set_title('Numerical One-Token')
axs[0,1].errorbar(numerical_cot.index, numerical_cot, yerr=np.sqrt(numerical_cot*(1-numerical_cot)/150), fmt='o', linestyle='', capsize=5, c='k')
axs[0,1].set_title('Numerical COT')
axs[1,0].errorbar(verbal_onetoken.index, verbal_onetoken, yerr=np.sqrt(verbal_onetoken*(1-verbal_onetoken)/150), fmt='o', linestyle='', capsize=5, c='k')
axs[1,0].set_title('Verbal One-Token')
axs[1,1].errorbar(verbal_cot.index, verbal_cot, yerr=np.sqrt(verbal_cot*(1-verbal_cot)/150), fmt='o', linestyle='', capsize=5, c='k')
axs[1,1].set_title('Verbal COT')
plt.savefig('cpc_validation_fig1.png')

# group by context and color-code

df_numerical.columns = ['context', 'confidence', 'one-token', 'cot']
df_numerical.set_index('context', inplace=True)
df_verbal.columns = ['context', 'confidence', 'one-token', 'cot']
df_verbal.set_index('context', inplace=True)

numerical_onetoken = df_numerical.groupby([df_numerical.index, 'confidence'])['one-token'].mean().reset_index(name='Stepback Rate')
numerical_cot = df_numerical.groupby([df_numerical.index, 'confidence'])['cot'].mean().reset_index(name='Stepback Rate')
verbal_onetoken = df_verbal.groupby([df_verbal.index, 'confidence'])['one-token'].mean().reset_index(name='Stepback Rate')
verbal_cot = df_verbal.groupby([df_verbal.index, 'confidence'])['cot'].mean().reset_index(name='Stepback Rate')

figs, axs = plt.subplots(2, 2, figsize=(10, 10))
for row in axs: 
    for ax in row:
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Stepback Rate')
        ax.set_ylim([0, 1])

axs[0,0].scatter(numerical_onetoken['confidence'], numerical_onetoken['Stepback Rate'], c=numerical_onetoken['context'])
axs[0,0].set_title('Numerical One-Token')
axs[0,1].scatter(numerical_cot['confidence'], numerical_cot['Stepback Rate'], c=numerical_cot['context'])
axs[0,1].set_title('Numerical COT')
axs[1,0].scatter(verbal_onetoken['confidence'], verbal_onetoken['Stepback Rate'], c=verbal_onetoken['context'])
axs[1,0].set_title('Verbal One-Token')
axs[1,1].scatter(verbal_cot['confidence'], verbal_cot['Stepback Rate'], c=verbal_cot['context'])
axs[1,1].set_title('Verbal COT')
plt.savefig('cpc_validation_fig2.png')