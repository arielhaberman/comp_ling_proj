import os
import json
import pandas as pd
from llama2 import Llama2Model

def process_tsv_files(data_dir, output_dir, model, max_history_len=3):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each TSV file in the specified directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.tsv'):
            # Load TSV file
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path, sep='\t')

            # Initialize conversation history
            history = []

            # Prepare output file
            output_path = os.path.join(output_dir, filename.replace('.tsv', '_results.tsv'))
            results = []

            # Process each line in the TSV file
            for _, row in df.iterrows():
                # Build observation from TSV row, e.g., `text` could be the student input
                observation = {"text": row['text']}

                # Generate a response using Llama2Model with history
                response = model.act(observation, history)

                # Store results and update history
                results.append({"text": row['text'], "response": response})
                history.insert(0, {"text": row['text'], "labels": [response]})
                if len(history) > max_history_len:
                    history.pop()

            # Save the results for this file
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, sep='\t', index=False)
            print(f"Processed and saved results for {filename}")

def main():
    # Load model configuration
    config_path = 'src/parlai/opts/gpt3.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize Llama2Model
    model = Llama2Model(config_path=config_path)

    # Define input and output directories (can be replaced with command-line arguments as needed)
    data_dir = 'data/0_datasets/tscc/'
    output_dir = 'results/'

    # Process all TSV files in the directory
    process_tsv_files(data_dir, output_dir, model)

if __name__ == "__main__":
    main()

#python -m src.parlai.scripts.run -m src.parlai.models.llama2:Llama2Model -o src/parlai/opts/gpt3.json -t TSCC -d data/0_datasets/tscc/ -O results/
