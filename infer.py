import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from openai import OpenAI
from transformers.generation.streamers import TextStreamer

max_seq_length = 4000  
load_in_4bit = True

def load_model(model_path, chat_template, r=16, lora_alpha=32, peft_path=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(model,
        r = r,                                                      # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = 0,                                           # Supports any, but = 0 is optimized
        bias = "none",                                              # Supports any, but = "none" is optimized
        use_gradient_checkpointing = True,
        random_state = 3407
    )
    
    tokenizer = get_chat_template(tokenizer, chat_template = chat_template)
    return model, tokenizer

def inf_oai(model, prompts):
    client = OpenAI()
    
    all_responses = []

    for prompt in tqdm(prompts):
        completion = client.chat.completions.create(
            model=model,
            messages=[
            {"role": "user", "content": prompt}
            ]
        )
        all_responses.append(completion.choices[0].message.content)
        
    return all_responses

def inf(model, tokenizer, prompts, batch_size=100):
    FastLanguageModel.for_inference(model)
    # text_streamer = TextStreamer(tokenizer)
    
    all_responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        print("\tgetting prompts")
        chat_prompts = [[{"role": "user", "content": p}] for p in batch_prompts]
        
        print("\tapplying chat template")
        inputs = tokenizer.apply_chat_template(
            chat_prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length
        ).to(model.device)
        # print(f"DEVICE: {model.device}")
        print("\tgenerating")
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_seq_length - inputs.shape[1],
            # streamer=text_streamer,
            use_cache=False,
            temperature=0.1,
            # temperature=0.0,
            min_p=0.1,
            pad_token_id=tokenizer.pad_token_id
        )
        
        print("\tdecoding")
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_responses.extend(batch_responses)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_responses

