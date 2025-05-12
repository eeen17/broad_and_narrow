import torch
from unsloth import FastLanguageModel
from typing import List, Union
import math

max_seq_length = 2048  
load_in_4bit = True

def load_model(model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def inf(model, tokenizer, prompts, batch_size=100):
   
    all_responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        chat_prompts = [[{"role": "user", "content": p}] for p in batch_prompts]
        
        inputs = tokenizer.apply_chat_template(
            chat_prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
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

