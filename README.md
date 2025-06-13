# broad_and_narrow

_________________________________________________________________

data gen (ft)
```
python sample_pipe.py --base {base} --n_digits {2} --n_samples {1000}
```

eval
```
python eval_pipe.py --base {base} --cot {True} --model_name {unsloth/Phi-4 or ftpath like outputs/test_run_8_cot_True_n_digits_2/checkpoint-60} --size {250}
```

train sft
```
python ft_lora.py --base {base} --model_path {unsloth/Phi-4} --cot {include or exclude} --data_file {ft_data/data_ft_{base}_{n_digits}.txt}
```
