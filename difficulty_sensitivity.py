import pandas as pd
import matplotlib.pyplot as plt

def load_and_analyse(file_path):

    data = pd.read_csv(file_path)

    # changing 'did_switch' to boolean for consistency
    data['did_switch'] = data['did_switch'].apply(lambda x: True if x == 'Yes' else False)

    # rate of switching
    switch_rates = data.groupby('difficulty')['did_switch'].mean()

    # rate of agreement between one-token and CoT results
    data['agreement'] = (data['one_token_cpc_result'] == data['cot_cpc_result']).astype(int)
    agreement_rates = data.groupby('difficulty')['agreement'].mean()

    return switch_rates, agreement_rates

def main():
    gpt_path = 'CPC_stepping_back/results/gpt3_experiment2.csv'
    gpt4_path = 'CPC_stepping_back/results/gpt4_experiment2.csv'


    gpt_rates, gpt_agreements = load_and_analyse(gpt_path)
    gpt4_rates, gpt4_agreements = load_and_analyse(gpt4_path)

    # Strategy Switch Rate
    plt.figure(figsize=(12, 6))
    plt.plot(gpt_rates.index, gpt_rates.values, label='GPT', marker='o')
    plt.plot(gpt4_rates.index, gpt4_rates.values, label='GPT-4', marker='o')
    plt.title('Strategy Switch Rate by Problem Difficulty')
    plt.xlabel('Problem Difficulty')
    plt.ylabel('Rate of Strategy Switch')
    plt.legend()
    plt.grid(True)
    plt.savefig('CPC_stepping_back/results/plot_strategy_switch_rate.png', format='png')
    plt.show()
    print("Plot saved as 'plot_strategy_switch_rate.png' in 'results'.")

    # Agreement Rate
    plt.figure(figsize=(12, 6))
    plt.plot(gpt_agreements.index, gpt_agreements.values, label='GPT', marker='o')
    plt.plot(gpt4_agreements.index, gpt4_agreements.values, label='GPT-4', marker='o')
    plt.title('Agreement Rate between One-Token and CoT Results by Problem Difficulty')
    plt.xlabel('Problem Difficulty')
    plt.ylabel('Agreement Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('CPC_stepping_back/results/plot_agreement_rate.png', format='png')
    plt.show()
    print("Plot saved as 'plot_agreement_rate.png' in 'results'.")

if __name__ == "__main__":
    main()
