import pandas as pd
import matplotlib.pyplot as plt

df_numerical = pd.read_csv('cpc_validation_results_numerical.csv')
df_verbal = pd.read_csv('cpc_validation_results_verbal.csv')

df_numerical.columns = ['context', 'confidence', 'one-token', 'cot']
df_numerical.set_index('context', inplace=True)
df_verbal.columns = ['context', 'confidence', 'one-token', 'cot']
df_verbal.set_index('context', inplace=True)

figs, axs = plt.subplots(2, 2, figsize=(10, 10))

numerical_onetoken = df_numerical.groupby([df_numerical.index, 'confidence'])['one-token'].mean().reset_index(name='Stepback Rate')
numerical_cot = df_numerical.groupby([df_numerical.index, 'confidence'])['cot'].mean().reset_index(name='Stepback Rate')
verbal_onetoken = df_verbal.groupby([df_verbal.index, 'confidence'])['one-token'].mean().reset_index(name='Stepback Rate')
verbal_cot = df_verbal.groupby([df_verbal.index, 'confidence'])['cot'].mean().reset_index(name='Stepback Rate')

for row in axs: 
    for ax in row:
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Stepback Rate')
        
axs[0,0].scatter(numerical_onetoken['confidence'], numerical_onetoken['Stepback Rate'], c=numerical_onetoken['context'])
axs[0,0].set_title('Numerical One-Token')
axs[0,1].scatter(numerical_cot['confidence'], numerical_cot['Stepback Rate'], c=numerical_cot['context'])
axs[0,1].set_title('Numerical COT')
axs[1,0].scatter(verbal_onetoken['confidence'], verbal_onetoken['Stepback Rate'], c=verbal_onetoken['context'])
axs[1,0].set_title('Verbal One-Token')
axs[1,1].scatter(verbal_cot['confidence'], verbal_cot['Stepback Rate'], c=verbal_cot['context'])
axs[1,1].set_title('Verbal COT')

plt.savefig('cpc_validation_figs.png')
plt.show()