import os
import json
import itertools
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

                if turn.role == "student":
                    # Add student turn to history with an empty placeholder for teacher response
                    student_text = turn.edited or turn.anonymised
                    history.append({"text": student_text, "labels": [""]})

                elif turn.role == "teacher":
                    # Check if the previous entry in history is a student without a response
                    if history and history[-1]["labels"] == [""]:
                        # Store the original teacher response for the last student entry
                        teacher_response = turn.edited or turn.anonymised
                        history[-1]["labels"] = [teacher_response]  # Update history with teacher's response

                # Generate prompt and response only if the last entry is a student awaiting a teacher response
                if history and history[-1]["labels"] != [""]:
                    # Capture the original teacher response before generating the Llama response
                    original_teacher_response = history[-1]["labels"][0]
                    observation = {"text": history[-1]["text"]}
                    
                    # Generate the prompt based on the current history
                    full_prompt = model.make_prompt(observation, history)
                    llama_response = model.generate_response(full_prompt)  # Generate Llama response only

                    # Save the prompt, original response, and Llama response separately
                    dpo_records.append({
                        "prompt": full_prompt,
                        "original_response": original_teacher_response,
                        "llama_response": llama_response
                    })
                    print(f"Full prompt:\n{full_prompt}")
                    print(f"Original response: {original_teacher_response}")
                    print(f"Llama response: {llama_response}")

                # Keep history to the last 3 entries to manage memory
                if len(history) > 3:
                    history.pop(0)

                end_time = time.time()
                duration = end_time - start_time
                print(f"Processed prompt in {duration:.2f} seconds")

            # Save the collected DPO records to a JSONL file with separated fields
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
    model = Llama2Model(config_path=config_path)

    data_dir = 'data/tscc_split/tiny'
    output_dir = 'results/dpo_data/'

    process_and_save_dpo_data(data_dir, output_dir, model)

if __name__ == "__main__":
    main()

#python -m src.parlai.scripts.run -m src.parlai.models.llama2:Llama2Model -o src/parlai/opts/gpt3.json -t TSCC -d data/0_datasets/tscc/ -O results/
