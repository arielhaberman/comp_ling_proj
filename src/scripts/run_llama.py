import os
import json
import pandas as pd
from llama2 import Llama2Model

def process_and_save_dpo_data(data_dir, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
        if filename.endswith('.tsv'):
            file_path = os.path.join(data_dir, filename)
            tscc_chat = Chat.from_tsv(file_path)  # Load chat from TSV file

            dpo_records = []
            history = []

            # Iterate over dialogic pairs to structure conversation
            pair_func = _make_dialogic_pairs(role_first="student")
            turns_by_role = itertools.groupby(tscc_chat.turns, key=lambda t: t.role)
            turns_by_role = ((role, list(turns_grp)) for role, turns_grp in turns_by_role)
            turns_by_pair = itertools.groupby(turns_by_role, key=lambda rt_pair: pair_func(rt_pair[0]))

            for i, (__, grouper) in enumerate(turns_by_pair):
                roles, turns = zip(*grouper)
                turns = tuple(turns)

                # Check if we have a student-teacher pair
                if len(turns) == 2:
                    student_text = _concat_turns(turns[0], use_edits=True)
                    teacher_text = _concat_turns(turns[1], use_edits=True)
                    observation = {"text": student_text}
                    history.insert(0, {"text": student_text, "labels": [teacher_text]})
                elif len(turns) == 1 and roles[0] == "student":
                    # Handle case where student turn is not followed by a teacher turn
                    student_text = _concat_turns(turns[0], use_edits=True)
                    observation = {"text": student_text}
                    history.insert(0, {"text": student_text, "labels": [""]})

                # Generate prompt and response
                full_prompt, llama_response = model.converse(observation, history)
                dpo_records.append({
                    "prompt": full_prompt,
                    "original_response": teacher_text if len(turns) == 2 else "",
                    "llama_response": llama_response
                })
                print(full_prompt)

                # Maintain conversation history length
                if len(history) > 3:
                    history.pop()

            # Save to file
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
