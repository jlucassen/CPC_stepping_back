import matplotlib.pyplot as plt
import os
import itertools
import pandas as pd

def rearrange_indices(len):
    return [0, 1] + list(range(3, len)) + [2]

def plot_stepback_rates_with_confidence(file_path):
    df = pd.read_csv(file_path)
    results_dir = 'results_prompt_sensitivity'
    os.makedirs(results_dir, exist_ok=True)

    # Create a list of color-marker combinations
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    markers = ['o', '^', 's', 'p', '*', '+', 'x', 'd', '<', '>', '|']

    fig, axs = plt.subplots(1, 3, figsize=(24, 7))

    # Plotting One Token and CoT results
    for idx, (prompt_idx, group_data) in enumerate(df.groupby('Prompt Index for One Token')):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        one_token_stats = group_data.groupby('Confidence')['One Token Result'].agg(['mean', 'sem'])
        one_token_stats = one_token_stats.reindex(one_token_stats.index[rearrange_indices(len(one_token_stats.index))])
        axs[0].errorbar(one_token_stats.index, one_token_stats['mean'], yerr=one_token_stats['sem'], fmt=marker, color=color, capsize=5, label=f'Prompt {prompt_idx} One-Token')

    for idx, (prompt_idx, group_data) in enumerate(df.groupby('Prompt Index for CoT')):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        cot_stats = group_data.groupby('Confidence')['CoT Result'].agg(['mean', 'sem'])
        cot_stats = cot_stats.reindex(cot_stats.index[rearrange_indices(len(cot_stats.index))])
        axs[1].errorbar(cot_stats.index, cot_stats['mean'], yerr=cot_stats['sem'], fmt=marker, color=color, capsize=5, label=f'Prompt {prompt_idx} CoT')

    # Compute pairwise accuracy for all combinations
    pairwise_accuracies = []
    token_indices = df['Prompt Index for One Token'].unique()
    cot_indices = df['Prompt Index for CoT'].unique()
    confidences = df['Confidence'].unique()

    for idx, (token_idx, cot_idx) in enumerate(itertools.product(token_indices, cot_indices)):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        for conf in confidences:
            token_conf_data = df[(df['Prompt Index for One Token'] == token_idx) & (df['Confidence'] == conf)]
            cot_conf_data = df[(df['Prompt Index for CoT'] == cot_idx) & (df['Confidence'] == conf)]

            if len(token_conf_data) > 0 and len(cot_conf_data) > 0:
                # Ensure the arrays have the same shape before calculating accuracy
                min_len = min(len(token_conf_data), len(cot_conf_data))
                token_conf_data = token_conf_data.iloc[:min_len]
                cot_conf_data = cot_conf_data.iloc[:min_len]

                accuracy = (token_conf_data['One Token Result'].values == cot_conf_data['CoT Result'].values).mean()
                pairwise_accuracies.append({'Confidence': conf, 'Pairwise Accuracy': accuracy, 'Color': color, 'Marker': marker, 'Label': f'1T: {token_idx}, CoT: {cot_idx}'})

    pairwise_df = pd.DataFrame(pairwise_accuracies)
    for label, group_data in pairwise_df.groupby('Label'):
        axs[2].plot(group_data['Confidence'], group_data['Pairwise Accuracy'], marker=group_data['Marker'].iloc[0], color=group_data['Color'].iloc[0], label=label)

    for ax in axs:
        ax.set_ylim(0, 1)
        ax.set_title(('Stepback Rate' if ax != axs[2] else 'Pairwise Accuracy') + ', ' + ('gpt-3.5-turbo' if '4' not in file_path else 'gpt-4') + ', ' + file_path.split('_')[-1][:-4])
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Rate' if ax != axs[2] else 'Accuracy')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plot_filename = os.path.splitext(os.path.basename(file_path))[0] + '_results.png'
    plt.savefig(os.path.join(results_dir, plot_filename), bbox_inches='tight')
    plt.close(fig)

results_dir = 'results_prompt_sensitivity'
files = ['cpc_validation_spoonfeed.csv', 'cpc_validation_batna.csv', 'cpc_validation_hints.csv', 'cpc_validation_red_herrings.csv']
for file in [results_dir+'/'+x for x in files]:
    plot_stepback_rates_with_confidence(file)
for file in [results_dir+'/4_'+x for x in files]:
    plot_stepback_rates_with_confidence(file)