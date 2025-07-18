import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from arithmetic.eval import get_label
from infer import inf, load_model

def load_data(data_file, size):
    x = [line.strip() for line in open(data_file)][:size]
    print(len(x))
    return x


def answer(expr, base):
    lhs, rhs = expr.split("+")
    lt, lo = lhs  # tens, ones
    rt, ro = rhs
    ones_sum = get_label(f"{lo}+{ro}", base)
    carry_over = len(ones_sum) > 1
    tens_sum_wo_carry = get_label(f"{lt}+{rt}", base)
    if carry_over:
        assert ones_sum[0] == "1"
        tens_sum_w_carry = get_label(f"{tens_sum_wo_carry}+1", base)
    else:
        tens_sum_w_carry = tens_sum_wo_carry
    assert get_label(expr, base) == tens_sum_w_carry + ones_sum[-1:]

    ret = f"We add the ones digits first. In base-{base}, {lo}+{ro}={ones_sum}. So the ones digit of the final sum is {ones_sum[-1:]}. "
    if carry_over:
        ret += f"We need to carry over the 1 to the tens place. "
    else:
        ret += f"We do not need to carry any digits over. "
    ret += f"Then we add the tens digits. In base-{base}, {lt}+{rt}={tens_sum_wo_carry}. "
    if carry_over:
        ret += f"Since we carried over the 1, {tens_sum_wo_carry}+1={tens_sum_w_carry}. "
    if len(tens_sum_w_carry) == 1:
        ret += f"So the tens digit of the final sum is {tens_sum_w_carry}. "
    else:
        ret += f"So the hundreds and tens digits of the final sum are {tens_sum_w_carry}. "
    ret += f"Putting the digits of the final sum together, we get \\boxed{{{tens_sum_w_carry}{ones_sum[-1:]}}}."
    return ret


def templatize(expr, base, cot=True, n_shots=0, icl_cot=True):
    if n_shots > 0:
        with open(f"ft_data/data_ft_{base}_2.txt",'r') as f:
            demos = f.read()
        shots = demos.split("\n")[:n_shots]
        assert len(shots) == n_shots
        if icl_cot:
            context = "\n".join(f"{templatize(shot, base)} {answer(shot, base)}" for shot in shots)
        else:
            context = "\n".join(f"{templatize(shot, base)} \\boxed{{{get_label(shot, base)}}}" for shot in shots)
        return "Examples: " + context + "\n" + templatize(expr, base)
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if cot:
        return f" You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? Let's think step by step, and end the response with the result in \"\\boxed{{result}}\"."
    else:
        return f" You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? End the response with the result in \"\\boxed{{result}}\"."


def escape(str):
    assert "\t" not in str and "\\\\n" not in str and "\\\\r" not in str
    return str.replace("\\n", "\\\\n").replace("\n", "\\n").replace("\\r", "\\\\r").replace("\r", "\\r")


def parse_bool(flag):
    if isinstance(flag, bool):
        return flag
    assert flag in {"True", "False"}
    return flag == "True"

import random

def main(data_file, base, model_name, output_file, cot=True, n_shots=0, size=250, icl_cot=True):
    base = int(base)
    cot = parse_bool(cot)
    n_shots = int(n_shots)
    data = load_data(data_file, size)

    assert not os.path.exists(output_file)
    print(base)
    templatized = [templatize(expr, base, cot=cot, n_shots=n_shots, icl_cot=icl_cot) for expr in data]

    model, tokenizer = load_model(model_name)
    responses = inf(model, tokenizer, templatized)

    with open(output_file, "w") as log:
        for expr, response in zip(data, responses, strict=True):
            log.write(f"{expr}\t{escape(response)}\n")


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter,too-many-function-args
    except Exception as e:
        import pdb
        import traceback

        if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()
