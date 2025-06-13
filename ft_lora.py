import torch
import os

from unsloth import FastLanguageModel 
from unsloth.chat_templates import standardize_sharegpt, get_chat_template
from openai import OpenAI

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
import argparse


max_seq_length = 2048  
load_in_4bit = True

def load_model(model_path, chat_template, r=16, lora_alpha=32, peft_path=None):
    model, tokenizer = FastLanguageModel.from_pretrained(model_name = model_path,
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
def get_dataset(base, cot, n_digits, data_file=None, oai=False):
    if data_file is None:
        data_file = f'ft_data/data_ft_{base}_{n_digits}.txt'
    x = [line.strip() for line in open(data_file)]
    if cot:
        y = [answer(line, base) for line in x]
    else:
        y = ['\\boxed{'+ str(get_label(line, base))+'}' for line in x]
    x = [templatize(line, base, cot) for line in x]
    m_head = "conversations"
    if oai:
        m_head = "messages"
    if not oai:
        with open('tmp.json', 'w') as f:
            json.dump([{"conversations":[{'role':'user','content':q},{'role':'assistant','content':a}]} for q,a in zip(x,y)], f)
            dataset = load_dataset('json', data_files='tmp.json')
    else:
        with open('tmp.jsonl', 'w') as f:
            for q, a in zip(x, y):
                json.dump({"messages":[{'role':'user','content':q},{'role':'assistant','content':a}]}, f)
                f.write("\n")
        dataset = None
    return dataset


def train_model(model, tokenizer, base, cot, n_digits,data_file,res_only=True):
    dataset = get_dataset(base, cot, n_digits, data_file)
    dataset = prep_dataset(dataset, tokenizer)
    
    name = f"test_run_{base}_cot_{cot}_n_digits_{n_digits}"  
    print(cot)
    print(dataset['train'][0])
    print(name)


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset['train'],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1, 
            learning_rate = 1e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs/" + name,
            save_strategy = "steps",
            save_steps = 5
        ),
    )

    if res_only:
        trainer = train_on_responses_only(trainer,
        instruction_part="<|im_start|>user<|im_sep|>",
        response_part="<|im_start|>assistant<|im_sep|>")
    
    trainer_stats = trainer.train()
    return trainer_stats


def train_model_oai(model, base, cot, n_digits,data_file):
  dataset = get_dataset(base, cot, n_digits, data_file, oai=True)
  client = OpenAI()
  
  f = client.files.create(
    file=open('tmp.jsonl', "rb"),
    purpose="fine-tune"
  )

  m = client.fine_tuning.jobs.create(
      training_file=f.id,
      model=model,
      hyperparameters = {'n_epochs':1, 'batch_size':16}
  )
  
  with open(f"outputs/oai_models", "a") as f:
      f.write(f"test_run_{base}_cot_{cot}_n_digits_{n_digits}"  + ": " + str(m.id))
  
 

def main(args):
    print(args)
    base = args.base
    model_path = args.model_path
    cot = args.cot
    n_digits = args.n_digits
    chat_template = "phi-4"
    r, lora_alpha = 64, 128

    if "gpt" not in args.model_path:
        model, tokenizer = load_model(model_path, chat_template, r, lora_alpha)
        print("training model...")
        train_stats = train_model(model, tokenizer, base, cot, n_digits, data_file=args.data_file, res_only=True)
    else:
        train_stats = train_model_oai(args.model_path, base, cot, n_digits, data_file=args.data_file)
    print(train_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--model_path", type=str, default="unsloth/Phi-4")
    parser.add_argument("--cot", action='store_true', default=False)
    parser.add_argument("--n_digits", type=int, default=2)
    parser.add_argument("--data_file", type=str, default=None)
    args = parser.parse_args()
    main(args)
