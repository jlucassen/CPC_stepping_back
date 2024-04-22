import matplotlib.pyplot as plt
import os
import itertools
import pandas as pd

def plot_stepback_rates_with_confidence(file_path):
    df = pd.read_csv(file_path)
    
    if 'knapsack' in file_path:
        df['Confidence'] = df['Confidence'].apply(lambda x: int(x * 10))
        xtick_labels = [f'resources expended: {i}' for i in range(1, 11)]
    else:
        df['Confidence'] = df['Confidence'].astype(str).str.extract(r'with (\d+)% probability').astype(int)
        xtick_labels = [f'with {i}% probability' for i in range(0, 101, 10)]

    df = df.sort_values('Confidence')

    results_dir = 'results_prompt_sensitivity'
    graphs_dir = os.path.join(results_dir, 'graphs_4_prompts')
    os.makedirs(graphs_dir, exist_ok=True)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    markers = ['o', '^', 's', 'p', '*', '+', 'x', 'd', '<', '>', '|']
    color_marker_combinations = list(itertools.product(colors, markers))

    fig, axs = plt.subplots(1, 3, figsize=(24, 7))

    # One Token and CoT results
    for idx, (prompt_idx, group_data) in enumerate(df.groupby('Prompt Index for One Token')):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        one_token_stats = group_data.groupby('Confidence', observed=True)['One Token Result'].agg(['mean', 'sem'])
        one_token_stats = one_token_stats.reindex(df['Confidence'].unique()).fillna(0)
        
        axs[0].errorbar(range(len(xtick_labels)), one_token_stats['mean'], yerr=one_token_stats['sem'], fmt=marker, color=color, capsize=5, label=f'Prompt {prompt_idx} One-Token')

    for idx, (prompt_idx, group_data) in enumerate(df.groupby('Prompt Index for CoT')):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        cot_stats = group_data.groupby('Confidence', observed=True)['CoT Result'].agg(['mean', 'sem'])
        cot_stats = cot_stats.reindex(df['Confidence'].unique()).fillna(0)
        
        axs[1].errorbar(range(len(xtick_labels)), cot_stats['mean'], yerr=cot_stats['sem'], fmt=marker, color=color, capsize=5, label=f'Prompt {prompt_idx} CoT')

    # Pairwise accuracy 
    pairwise_accuracies = []
    token_indices = df['Prompt Index for One Token'].unique()
    cot_indices = df['Prompt Index for CoT'].unique()

    for idx, (token_idx, cot_idx) in enumerate(itertools.product(token_indices, cot_indices)):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        for conf in df['Confidence'].unique():
            token_conf_data = df[(df['Prompt Index for One Token'] == token_idx) & (df['Confidence'] == conf)]
            cot_conf_data = df[(df['Prompt Index for CoT'] == cot_idx) & (df['Confidence'] == conf)]

            if len(token_conf_data) > 0 and len(cot_conf_data) > 0:
                min_len = min(len(token_conf_data), len(cot_conf_data))
                token_conf_data = token_conf_data.iloc[:min_len]
                cot_conf_data = cot_conf_data.iloc[:min_len]

                accuracy = (token_conf_data['One Token Result'].values == cot_conf_data['CoT Result'].values).mean()
                pairwise_accuracies.append({'Confidence': conf, 'Pairwise Accuracy': accuracy, 'Color': color, 'Marker': marker, 'Label': f'1T: {token_idx}, CoT: {cot_idx}'})

    pairwise_df = pd.DataFrame(pairwise_accuracies)
    pairwise_df = pairwise_df.pivot(index='Confidence', columns='Label', values='Pairwise Accuracy').reindex(df['Confidence'].unique()).reset_index()

    for idx, label in enumerate(pairwise_df.columns[1:]):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        axs[2].plot(range(len(xtick_labels)), pairwise_df[label], marker=marker, color=color, label=label)

    for ax in axs[:2]:
        ax.set_ylim(0, 1)
        ax.set_title('Stepback Rate')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Rate')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=90)

    axs[2].set_ylim(0, 1)
    axs[2].set_title('Pairwise Accuracy')
    axs[2].set_xlabel('Confidence')
    axs[2].set_ylabel('Accuracy')
    axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[2].set_xticks(range(len(xtick_labels)))
    axs[2].set_xticklabels(xtick_labels, rotation=90)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plot_filename = os.path.splitext(os.path.basename(file_path))[0] + '_results_corrected.png'
    plt.savefig(os.path.join(graphs_dir, plot_filename), bbox_inches='tight')
    plt.close(fig)


files = [
    '4_cpc_validation_spoonfeed.csv', 
    '4_cpc_validation_batna.csv', 
    '4_cpc_validation_hints.csv', 
    '4_cpc_validation_knapsack.csv', 
    '4_cpc_validation_red_herrings.csv'
]


for file in files:
    file_path = os.path.join('results_prompt_sensitivity', file)
    plot_stepback_rates_with_confidence(file_path)