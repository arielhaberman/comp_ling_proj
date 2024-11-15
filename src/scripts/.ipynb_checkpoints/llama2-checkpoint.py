# llama2.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re

# Define constants for the student-teacher dialogue
STUDENT_PREFIX = 'Student:'
TEACHER_PREFIX = 'Teacher:'
AI_PREFIX = 'AI:'
STOP = ["\n", f"{STUDENT_PREFIX}", f"{TEACHER_PREFIX}", f"{AI_PREFIX}"]
INSTRUCTIONS = "The following is a conversation with a teacher. Please continue the conversation as the teacher."

class Llama2Model:
    MAX_CONTEXT_LEN = 4096  # Example max token length for Llama 2

    def __init__(self, config_path="gpt3.json"):
        # Load settings from configuration file
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.max_tokens = config.get("gpt3_max_tokens", 500)
        self.temperature = config.get("gpt3_temperature", 0.7)
        model_path = "/scratch/ah5192/comp_ling/models/llama-2-7b-chat-hf"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    @classmethod
    def restrict_history(cls, context, history, turn, max_history_len=None, max_completion_len=0):
        max_prompt_len = cls.MAX_CONTEXT_LEN - max_completion_len
        count = len(cls.tokenize(context)) + len(cls.tokenize(turn)) + 2

        for h, hist in enumerate(history[:max_history_len]):
            q, a = hist.get('text', ''), hist.get('labels', [''])[0]
            q_len, a_len = len(cls.tokenize(q)), len(cls.tokenize(a))
            if (q or a) and count + q_len + a_len + 2 < max_prompt_len:
                count += q_len + a_len + 2
                yield q, a
            else:
                break

    @staticmethod
    def make_pair(turn, answer, prefix_t=STUDENT_PREFIX, prefix_ai=TEACHER_PREFIX):
        pair = [f"{prefix_t} {turn}" if turn else "", f"{prefix_ai} {answer}" if answer else ""]
        return '\n'.join(pair).strip()

    @classmethod
    def make_prompt(cls, observation, history, instructions=INSTRUCTIONS, prefix_t=STUDENT_PREFIX, prefix_ai=TEACHER_PREFIX):
        # Build prompt from history
        history_res = []
        for entry in history:
            student_text = entry['text']
            teacher_text = entry['labels'][0] if entry['labels'][0] else ""
            history_res.append(f"Student: {student_text}\nTeacher: {teacher_text}")
    
        history_text = "\n\n".join(history_res)
    
        # Add "Teacher:" at the end only if the last entry is a student with no teacher response
        # if history and history[-1]["labels"] == [""]:
        #     prompt = f"{instructions}\n\n{history_text}\nTeacher:"
        # else:
        prompt = f"{instructions}\n\n{history_text}"
    
        return prompt

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_tokens + inputs["input_ids"].shape[1],
                do_sample=True,
                temperature=self.temperature,
                eos_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def converse(self, observation, history):
        # Generate prompt with restricted history
        prompt = self.make_prompt(observation, history)
        
        response = self.generate_response(prompt)
        return prompt, response  # Return full prompt and response


    @staticmethod
    def tokenize(text):
        # Simulate token count (use tokenizer if counting tokens)
        return text.split()

