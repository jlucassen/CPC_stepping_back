import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_stepback_rates_with_confidence(file_path):
    df = pd.read_csv(file_path)
    
    one_token_stats = df.groupby('Confidence')['One Token Result'].agg(['mean', 'sem'])
    cot_stats = df.groupby('Confidence')['CoT Result'].agg(['mean', 'sem'])
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    axs[0].errorbar(one_token_stats.index, one_token_stats['mean'], yerr=one_token_stats['sem'], fmt='o', capsize=5, label='One-Token')
    axs[0].set_title('One-Token Stepback Rate')
    axs[0].set_xlabel('Confidence')
    axs[0].set_ylabel('Stepback Rate')
    axs[0].set_ylim(0, 1)

    axs[1].errorbar(cot_stats.index, cot_stats['mean'], yerr=cot_stats['sem'], fmt='o', capsize=5, label='CoT')
    axs[1].set_title('CoT Stepback Rate')
    axs[1].set_xlabel('Confidence')
    axs[1].set_ylabel('Stepback Rate')
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_filename}_stepback_rate_comparison.png"
    plt.savefig(output_filename)
    plt.close()

plot_stepback_rates_with_confidence('results_prompt_sensitivity/cpc_validation_spoonfeed.csv')
plot_stepback_rates_with_confidence('results_prompt_sensitivity/cpc_validation_spoonfeed.csv')
plot_stepback_rates_with_confidence('results_prompt_sensitivity/cpc_validation_batna.csv')
plot_stepback_rates_with_confidence('results_prompt_sensitivity/cpc_validation_hints.csv')
plot_stepback_rates_with_confidence('results_prompt_sensitivity/cpc_validation_knapsack.csv')
plot_stepback_rates_with_confidence('results_prompt_sensitivity/cpc_validation_red_herrings.csv')
