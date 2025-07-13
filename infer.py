import torch
import os
from typing import List, Union
from unsloth import FastLanguageModel 
from unsloth.chat_templates import standardize_sharegpt, get_chat_template
from unsloth.chat_templates import train_on_responses_only
import argparse
import math

load_in_4bit = True
max_seq_length = 10000
def load_model(model_path, chat_template = "phi-4", r=16, lora_alpha=32, peft_path=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 10000,
        load_in_4bit = load_in_4bit,
        # token = hf_
    )
    tokenizer = get_chat_template(tokenizer, chat_template = chat_template)
    return model, tokenizer


def inf(model, tokenizer, prompts, batch_size=180):
    all_responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        chat_prompts = [[{"role": "user", "content": p}] for p in batch_prompts]
        
        inputs = tokenizer.apply_chat_template(
            chat_prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=10000,
        ).to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=2000,
                use_cache=True,
                
                pad_token_id=tokenizer.pad_token_id
            )
        
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_responses.extend(batch_responses)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_responses

