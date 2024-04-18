import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_stepback_rates_with_confidence(file_path):
    df = pd.read_csv(file_path)

    results_dir = 'results_prompt_sensitivity'
    os.makedirs(results_dir, exist_ok=True)
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', '^', 's', 'p', '*', '+', 'x', 'd']
    
    fig, axs = plt.subplots(1, 3, figsize=(24, 7))
    
    #  plots
    for (prompt_idx, group_data) in df.groupby('Prompt Index for One Token'):
        color = colors[prompt_idx % len(colors)]
        marker = markers[prompt_idx % len(markers)]
        one_token_stats = group_data.groupby('Confidence')['One Token Result'].agg(['mean', 'sem'])
        axs[0].errorbar(one_token_stats.index, one_token_stats['mean'], yerr=one_token_stats['sem'], fmt=marker, color=color, capsize=5, label=f'Prompt {prompt_idx} One-Token')
    
    for (prompt_idx, group_data) in df.groupby('Prompt Index for CoT'):
        color = colors[prompt_idx % len(colors)]
        marker = markers[prompt_idx % len(markers)]
        cot_stats = group_data.groupby('Confidence')['CoT Result'].agg(['mean', 'sem'])
        axs[1].errorbar(cot_stats.index, cot_stats['mean'], yerr=cot_stats['sem'], fmt=marker, color=color, capsize=5, label=f'Prompt {prompt_idx} CoT')
    
    # Pairwise accuracy calculation corrected
    pairwise_accuracy = df.groupby(
        ['Confidence', 'Prompt Index for One Token', 'Prompt Index for CoT']
    ).apply(lambda x: ((x['One Token Result'].values == x['CoT Result'].values).mean())).reset_index(name='Pairwise Accuracy')

    for (prompt_idx_1t, prompt_idx_cot), group_data in pairwise_accuracy.groupby(['Prompt Index for One Token', 'Prompt Index for CoT']):
        color = colors[prompt_idx_1t % len(colors)]
        marker = markers[prompt_idx_1t % len(markers)]
        axs[2].plot(group_data['Confidence'], group_data['Pairwise Accuracy'], marker, color=color, label=f'1T Prompt {prompt_idx_1t}, CoT Prompt {prompt_idx_cot}')

    for ax in axs:
        ax.set_ylim(0, 1)
        ax.set_title('Stepback Rate' if ax != axs[2] else 'Pairwise Accuracy')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Rate' if ax != axs[2] else 'Accuracy')
        ax.legend(loc='upper left', bbox_to_anchor=(1,1))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust for legend
    plot_filename = os.path.splitext(os.path.basename(file_path))[0] + '_results.png'
    plt.savefig(os.path.join(results_dir, plot_filename), bbox_inches='tight')
    plt.close(fig)


plot_stepback_rates_with_confidence('cpc_validation_spoonfeed.csv')
plot_stepback_rates_with_confidence('cpc_validation_batna.csv')
plot_stepback_rates_with_confidence('cpc_validation_hints.csv')
plot_stepback_rates_with_confidence('cpc_validation_red_herrings.csv')
plot_stepback_rates_with_confidence('cpc_validation_knapsack.csv')
