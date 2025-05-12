import torch

from unsloth import FastLanguageModel 
from unsloth.chat_templates import standardize_sharegpt, get_chat_template

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

from unsloth.chat_templates import train_on_responses_only


max_seq_length = 2048  
load_in_4bit = True

def load_model(model_path, chat_template, r=64, lora_alpha=128):
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
    dataset = standardize_sharegpt(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
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

def get_dataset(dataset_path):
    try:
        dataset = load_dataset(dataset_path)
    except:
        dataset = load_dataset('csv', data_files='my_file.csv')
    return dataset

def train_model(model, tokenizer, dataset_path, name, res_only=True):
    dataset = get_dataset(dataset_path)
    dataset = prep_dataset(dataset, tokenizer)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 30,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs/" + name,
            report_to = "none",
            save_strategy = "steps"
        ),
    )

    if res_only:
        trainer = train_on_responses_only(trainer,
        instruction_part="<|im_start|>user<|im_sep|>",
        response_part="<|im_start|>assistant<|im_sep|>")

    trainer_stats = trainer.train()
    return trainer_stats


def main():
    model_path =  "unsloth/Phi-4"
    chat_template = "phi-4"
    r, lora_alpha = 64, 128
    dataset_path = 
    name = "test_run"

    model, tokenizer = load_model(model_path, chat_template, r, lora_alpha)

    train_stats = train_model(model, tokenizer, dataset_path, name, res_only=True)
    print(train_stats)

main()

