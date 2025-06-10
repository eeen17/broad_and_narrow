import os

bases = [8, 9, 10]
cots = ["--cot", ""]
eval_n_digits = [2, 3, 4]
train_n_digits = [2]
train_types = ["ft_lora"]
eval_checkpoints = [60, 50, 40, 30, 20, 10]

command_file = "eval_commands.sh"
                
## eval code
models = os.listdir("outputs")
for checkpoint in eval_checkpoints:
    for base in bases:
        for n_digits in eval_n_digits:
            for model in models:
                command = f"python eval_pipe.py --base {base} --model_name outputs/{model}/checkpoint-{checkpoint} --size 300 --n_digits {n_digits}"
                with open(command_file, "a") as f:
                    f.write(command + "\n")
# and baseline evals
for base in bases:
    for n_digits in eval_n_digits:
        command = f"python eval_pipe.py --base {base} --model_name unsloth/Phi-4 --size 300 --n_digits {n_digits}"
        with open(command_file, "a") as f:
            f.write(command + "\n")
                
