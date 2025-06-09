# broad_and_narrow


data gen (ft)
```
python sample_pipe.py --base {base} --n_digits {2} --n_samples {1000}
```

eval
```
python eval_pipe.py --base {base} --cot {True} --model_name {unsloth/Phi-4 or ftpath} --size {200}
```

train sft
```
python ft_lora.py --base {base} --model_path {unsloth/Phi-4} --cot {T/F}
```
