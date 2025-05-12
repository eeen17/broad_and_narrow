import torch
from unsloth import FastLanguageModel

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

def inf(model, tokenizer, prompts):
    # prompts = [{"role": "user", "content": "Describe a tall tower in the capital of France."},]
    inputs = tokenizer.apply_chat_template(
        prompts,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to(model.device)

    outputs = model.generate(input_ids = inputs, max_new_tokens = 256, use_cache = True, temperature = 0.0, min_p = 0.1)
    out = tokenizer.batch_decode(outputs)

    return out

