import os
import json
import itertool
import time
import pandas as pd
from llama2 import Llama2Model
from tscc import Chat, _make_dialogic_pairs, _concat_turns

def process_and_save_dpo_data(data_dir, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)
    big_start = time.time()
    for filename in os.listdir(data_dir):
        if filename.endswith('.tsv'):
            file_path = os.path.join(data_dir, filename)
            tscc_chat = Chat.from_tsv(file_path)

            dpo_records = []
            history = []

            for turn in tscc_chat.turns:
                start_time = time.time()

                # Process each turn based on the role
                if turn.role == "student":
                    # Add student turn to history with no teacher response initially
                    student_text = turn.edited or turn.anonymised
                    history.append({"text": student_text, "labels": [""]})

                elif turn.role == "teacher":
                    # If there's an unpaired student in history, add teacher's response
                    if history and history[-1]["labels"] == [""]:
                        history[-1]["labels"] = [turn.edited or turn.anonymised]

                # Prepare observation only if the last entry is a student with no teacher response
                if history and history[-1]["labels"] == [""]:
                    observation = {"text": history[-1]["text"]}
                else:
                    observation = None

                if observation is not None:
                    # Generate prompt and response
                    full_prompt, llama_response = model.converse(observation, history)
                    dpo_records.append({
                        "prompt": full_prompt,
                        "original_response": history[-1]["labels"][0],
                        "llama_response": llama_response
                    })
                    print(full_prompt)

                # Maintain a maximum of 3 history entries
                if len(history) > 3:
                    history.pop(0)

                end_time = time.time()
                duration = end_time - start_time
                print(f"Processed prompt in {duration:.2f} seconds")

            output_path = os.path.join(output_dir, f"{filename.replace('.tsv', '_dpo.jsonl')}")
            with open(output_path, 'w') as f:
                for record in dpo_records:
                    f.write(json.dumps(record) + '\n')

            print(f"Processed and saved DPO data for {filename}")

    big_end = time.time()
    big_duration = big_end - big_start
    print(f"Processed files in {big_duration:.2f} seconds")

def main():
    config_path = 'src/opts/gpt3.json'
    model = Llama2Model(config_path=config_path).to("cuda")

    data_dir = 'data/tscc_split/train'
    output_dir = 'results/dpo_data/'

    process_and_save_dpo_data(data_dir, output_dir, model)

if __name__ == "__main__":
    main()

#python -m src.parlai.scripts.run -m src.parlai.models.llama2:Llama2Model -o src/parlai/opts/gpt3.json -t TSCC -d data/0_datasets/tscc/ -O results/
