# broad_and_narrow

## Setup
1. Install `uv`. [Link to docs](https://docs.astral.sh/uv/guides/install-python/).
2. When running code, `uv` will install all packages for you.

<br>



---
> If not using `cuda:0`, may have to edit [.venv/lib/python3.12/site-packages/unsloth/models/llama.py#L1000](.venv/lib/python3.12/site-packages/unsloth/models/llama.py#L1000)
- Comment out and replace like the following:
```python
def LlamaModel_fast_forward_inference_custom(...):
    ...
    device = self.model.device
    residual = torch.empty((bsz, q_len, hd), dtype = torch.float32, device = device)
    _XX = torch.empty((2, bsz, q_len, hd), dtype = torch.float32, device = device)
    # residual = torch.empty((bsz, q_len, hd), dtype = torch.float32, device = f"{DEVICE_TYPE}:0")
    # _XX = torch.empty((2, bsz, q_len, hd), dtype = torch.float32, device = f"{DEVICE_TYPE}:0")
    ...
    variance = torch.empty((bsz, q_len, 1), dtype = torch.float32, device = device)
    temp_mlp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = device)
    # variance = torch.empty((bsz, q_len, 1), dtype = torch.float32, device = f"{DEVICE_TYPE}:0")
    # temp_mlp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = f"{DEVICE_TYPE}:0")
```
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
