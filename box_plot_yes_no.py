import os
import csv
import matplotlib.pyplot as plt

def create_box_plot_from_csv(csv_filename, plot_filename):
    with open(csv_filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        results = list(reader)

    data = {}
    for result in results:
        prompt_variation = result['prompt phrasing']
        accuracy_rate = float(result['accuracy rate'])  # Convert accuracy rate to float
        if prompt_variation not in data:
            data[prompt_variation] = []
        data[prompt_variation].append(accuracy_rate)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data.values())
    ax.set_xticklabels(data.keys(), rotation=45, ha='right')
    ax.set_ylabel('Accuracy Rate')
    ax.set_title('Accuracy Rate by Prompt Variation')
    plt.tight_layout()
    plt.savefig(plot_filename)

if __name__ == "__main__":
    create_box_plot_from_csv('yes_no_results.csv', 'yes_no_accuracy_box_plot.png')
