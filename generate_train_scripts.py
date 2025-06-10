import os

os.makedirs("ft_data", exist_ok=True)

bases = [8, 9, 10]
cots = ["--cot True", ""]
eval_n_digits = [2, 3, 4]
train_n_digits = [2]
train_types = ["ft_lora"]
eval_checkpoints = [60, 50, 40, 30, 20, 10]

command_file = "train_commands.sh"


## datagen code
for base in bases:
    for n_digits in train_n_digits:    
        command = f"python sample_pipe.py --base {base} --n_digits {2} --n_samples {500}"
        with open(command_file, "a") as f:
            f.write(command + "\n")


## training code
for train_type in train_types:
    for base in bases:
        for cot in cots:
            for n_digits in train_n_digits:
                command = f"python {train_type}.py --base {base} --model_path unsloth/Phi-4 {cot} --data_file ft_data/data_ft_{base}_{n_digits}.txt"
                with open(command_file, "a") as f:
                    f.write(command + "\n")
   
