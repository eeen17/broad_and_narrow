# broad_and_narrow

## Setup
1. Install `uv`. [Link to docs](https://docs.astral.sh/uv/guides/install-python/).
2. When running code, `uv` will install all packages for you.

_________________________________________________________________
## Run

```bash
uv run eval_pipe.py --base 10 --cot True --model_name unsloth/llama-3-8b-bnb-4bit --size 250 --chat_template unsloth --device cuda:6
```

---
old

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

