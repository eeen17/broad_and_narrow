import torch
from unsloth import FastLanguageModel 
from unsloth.chat_templates import standardize_sharegpt, get_chat_template
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
import argparse
from peft import LoraConfig, get_peft_model
import numpy as np
from arithmetic.eval import get_label

max_seq_length = 2048  
load_in_4bit = True

def load_model(model_path, chat_template, r=64, lora_alpha=128, peft_path=None):
    model, tokenizer = FastLanguageModel.from_pretrained(model_name = model_path,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    return model, tokenizer

def compute_reward(predictions, references, base, original_expressions):
    rewards = []
    for pred, ref, expr in zip(predictions, references, original_expressions):
        try:
            # Extract the answer from the prediction
            pred_answer = pred.split("\\boxed{")[-1].split("}")[0]
            # Get the correct answer using the evaluation logic
            correct_answer = get_label(expr, base)
            
            # Compare answers
            if pred_answer == correct_answer:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        except:
            rewards.append(-1.0)
    return torch.tensor(rewards)

def prep_dataset(dataset, tokenizer):
    print(dataset)
    def formatting_prompts_func(examples):
        convos = examples['conversations']
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize = False, add_generation_prompt = False
            )
            for convo in convos
        ]
        return { "text" : texts, }

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True)
    return dataset

from arithmetic.query import templatize, answer
from arithmetic.eval import get_label
import json

def get_dataset(base, cot, n_digits, data_file=None):
    if data_file is None:
        data_file = f'ft_data/data_ft_{base}_{n_digits}.txt'
    x = [line.strip() for line in open(data_file)][:500]
    if cot:
        y = [answer(line, base) for line in x]
    else:
        y = ['\\boxed{'+ str(get_label(line, base))+'}' for line in x]
    x = [templatize(line, base, cot) for line in x]
    with open('tmp.json', 'w') as f:
        json.dump([{"conversations":[{'role':'user','content':q},{'role':'assistant','content':a}]} for q,a in zip(x,y)], f)
    dataset = load_dataset('json', data_files='tmp.json')
    # Store original expressions for reward computation
    dataset = dataset.map(lambda x: {"original_expression": x["conversations"][0]["content"].split("what is ")[-1].split("?")[0]})
    return dataset

def train_model(model, tokenizer, base, cot, n_digits, data_file, res_only=True):
    dataset = get_dataset(base, cot, n_digits, data_file)
    dataset = prep_dataset(dataset, tokenizer)
    
    name = f"grpo_run_{base}_cot_{cot}_n_digits_{n_digits}"  
    print(cot)
    print(dataset['train'][0])
    print(name)

    # PPO configuration
    ppo_config = PPOConfig(
        learning_rate=2e-5,
        batch_size=2,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
        early_stopping=True,
        target_kl=0.1,
        ppo_epochs=1,
        seed=3407,
        init_kl_coef=0.2,
        adap_kl_ctrl=True,
    )

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset['train'],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    )

    # Training loop
    for epoch in range(1):  # Number of epochs
        for batch in ppo_trainer.dataloader:
            # Generate responses
            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
            )
            
            # Decode responses
            responses = tokenizer.batch_decode(response_tensors)
            references = tokenizer.batch_decode(batch["labels"])
            original_expressions = batch["original_expression"]
            
            # Compute rewards using evaluation logic
            rewards = compute_reward(responses, references, base, original_expressions)
            
            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            
            # Save checkpoint
            if ppo_trainer.config.save_steps > 0 and ppo_trainer.step % ppo_trainer.config.save_steps == 0:
                ppo_trainer.save_pretrained(f"outputs/{name}/checkpoint-{ppo_trainer.step}")

    return ppo_trainer

def main(args):
    print(args)
    base = args.base
    model_path = args.model_path
    cot = args.cot
    print(cot)
    n_digits = args.n_digits
    chat_template = "phi-4"
    r, lora_alpha = 64, 128

    model, tokenizer = load_model(model_path, chat_template, r, lora_alpha)
    print("training model...")
    train_stats = train_model(model, tokenizer, base, cot, n_digits, data_file=args.data_file, res_only=True)
    print(train_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--model_path", type=str, default="unsloth/Phi-4")
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--n_digits", type=int, default=2)
    parser.add_argument("--data_file", type=str, default=None)
    args = parser.parse_args()
    main(args)