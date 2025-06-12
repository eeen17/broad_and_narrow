import torch
import os
from typing import List, Union
from unsloth import FastLanguageModel 
from unsloth.chat_templates import standardize_sharegpt, get_chat_template
from openai import OpenAI

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
import argparse
import math

max_seq_length = 4000  
load_in_4bit = True

def load_model(model_path, chat_template, r=64, lora_alpha=128, peft_path=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        # token = hf_
    )

    model = FastLanguageModel.get_peft_model(model,
        r = r, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = 0, 
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407
    )

    tokenizer = get_chat_template(tokenizer, chat_template = chat_template)
    return model, tokenizer

def inf_oai(model, prompts):
    client = OpenAI()
    
    all_responses = []

    for prompt in prompts:
        completion = client.chat.completions.create(
            model=model,
            messages=[
            {"role": "user", "content": prompt}
            ]
        )
        all_responses.append(completion.choices[0].message.content)
        
    return all_responses

def inf(model, tokenizer, prompts, batch_size=100):
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
            max_length=max_seq_length
        ).to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_seq_length - inputs.shape[1],
                use_cache=True,
                temperature=0.0,
                min_p=0.1,
                pad_token_id=tokenizer.pad_token_id
            )
        
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_responses.extend(batch_responses)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_responses

