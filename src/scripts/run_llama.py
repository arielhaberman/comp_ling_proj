import os
import json
import pandas as pd
from llama2 import Llama2Model

def process_and_save_dpo_data(data_dir, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
        if filename.endswith('.tsv'):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path, sep='\t')

            dpo_records = []  # Store DPO data in a list of dictionaries

            history = ""
            for _, row in df.iterrows():
                # Extract student-teacher conversation data
                student_text = row['text']
                original_response = row.get('response', '')  # Original response in TSV

                # Create prompt and get Llama response
                prompt = f"Student: {student_text}\nTeacher:"
                full_prompt, llama_response = model.converse(prompt, history)

                # Append to DPO records
                dpo_records.append({
                    "prompt": full_prompt,
                    "original_response": original_response,
                    "llama_response": llama_response
                })

                # Update history for the conversation
                history += f"{prompt} {original_response}\n"

            # Save DPO records to a JSONL file
            output_path = os.path.join(output_dir, f"{filename.replace('.tsv', '_dpo.jsonl')}")
            with open(output_path, 'w') as f:
                for record in dpo_records:
                    f.write(json.dumps(record) + '\n')

            print(f"Processed and saved DPO data for {filename}")

def main():
    config_path = 'src/opts/gpt3.json'
    model = Llama2Model(config_path=config_path)

    data_dir = 'data/tscc_split/train'
    output_dir = 'results/dpo_data/'

    process_and_save_dpo_data(data_dir, output_dir, model)

if __name__ == "__main__":
    main()

#python -m src.parlai.scripts.run -m src.parlai.models.llama2:Llama2Model -o src/parlai/opts/gpt3.json -t TSCC -d data/0_datasets/tscc/ -O results/
