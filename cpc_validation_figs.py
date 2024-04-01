import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_numerical = pd.read_csv('cpc_validation_results_numerical.csv', header=None, names=['context', 'confidence', 'one-token', 'cot'])
df_verbal = pd.read_csv('cpc_validation_results_verbal.csv', header=None, names=['context', 'confidence', 'one-token', 'cot'])

# figs without grouping by context

numerical_onetoken1 = df_numerical.groupby('confidence')['one-token'].mean()
numerical_cot1 = df_numerical.groupby('confidence')['cot'].mean()
verbal_onetoken1 = df_verbal.groupby('confidence')['one-token'].mean()
verbal_cot1 = df_verbal.groupby('confidence')['cot'].mean()

figs, axs = plt.subplots(2, 2, figsize=(10, 10))
for row in axs: 
    for ax in row:
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Stepback Rate')
        ax.set_ylim([0, 1])

numerical_sample_size = len(df_numerical.index)/len(numerical_onetoken1.index)
verbal_sample_size = len(df_verbal.index)/len(verbal_onetoken1.index)

axs[0,0].errorbar(numerical_onetoken1.index, numerical_onetoken1, yerr=np.sqrt(numerical_onetoken1*(1-numerical_onetoken1)/numerical_sample_size), fmt='o', linestyle='', capsize=5, c='k')
axs[0,0].set_title('Numerical One-Token')
axs[0,1].errorbar(numerical_cot1.index, numerical_cot1, yerr=np.sqrt(numerical_cot1*(1-numerical_cot1)/numerical_sample_size), fmt='o', linestyle='', capsize=5, c='k')
axs[0,1].set_title('Numerical COT')
axs[1,0].errorbar(verbal_onetoken1.index, verbal_onetoken1, yerr=np.sqrt(verbal_onetoken1*(1-verbal_onetoken1)/verbal_sample_size), fmt='o', linestyle='', capsize=5, c='k')
axs[1,0].set_title('Verbal One-Token')
axs[1,1].errorbar(verbal_cot1.index, verbal_cot1, yerr=np.sqrt(verbal_cot1*(1-verbal_cot1)/verbal_sample_size), fmt='o', linestyle='', capsize=5, c='k')
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