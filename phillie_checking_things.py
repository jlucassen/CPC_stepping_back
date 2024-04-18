def display_first_line_builtin(file_path):
    with open(file_path, 'r') as file:
        header = next(file).strip()  # Read the header line
        first_line = next(file).strip()  # Read the next line, which is the first row of data
    print("Header:", header)
    print("First data line:", first_line)

def main():
    gpt_path = 'CPC_stepping_back/results/gpt3_experiment2.csv'
    gpt4_path = 'CPC_stepping_back/results/gpt4_experiment2.csv'
    
    print("First line from GPT experiment data using Built-in methods:")
    display_first_line_builtin(gpt_path)
    
    print("\nFirst line from GPT-4 experiment data using Built-in methods:")
    display_first_line_builtin(gpt4_path)

if __name__ == "__main__":
    main()
