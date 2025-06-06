python ft_lora.py --base 8 --model_path unsloth/Phi-4 --cot True --data_file ft_data/data_ft_8_2.txt
python ft_lora.py --base 8 --model_path unsloth/Phi-4 --data_file ft_data/data_ft_8_2.txt
python ft_lora.py --base 9 --model_path unsloth/Phi-4 --cot True --data_file ft_data/data_ft_9_2.txt
python ft_lora.py --base 9 --model_path unsloth/Phi-4 --data_file ft_data/data_ft_9_2.txt
python ft_lora.py --base 10 --model_path unsloth/Phi-4 --cot True --data_file ft_data/data_ft_10_2.txt
python ft_lora.py --base 10 --model_path unsloth/Phi-4 --data_file ft_data/data_ft_10_2.txt
python ft_lora.py --base 11 --model_path unsloth/Phi-4 --cot True --data_file ft_data/data_ft_11_2.txt
python ft_lora.py --base 11 --model_path unsloth/Phi-4 --data_file ft_data/data_ft_11_2.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 9 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 10 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-50 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-40 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-30 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-20 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-10 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 11 --model_name unsloth/Phi-4 --size 300 --n_digits 2
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 8 --model_name unsloth/Phi-4 --size 300 --n_digits 3
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 8 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 8 --model_name unsloth/Phi-4 --size 300 --n_digits 4
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 9 --model_name unsloth/Phi-4 --size 300 --n_digits 3
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 9 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 9 --model_name unsloth/Phi-4 --size 300 --n_digits 4
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 10 --model_name unsloth/Phi-4 --size 300 --n_digits 3
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 10 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 10 --model_name unsloth/Phi-4 --size 300 --n_digits 4
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 3
rm output.txt
python eval.py --base 11 --model_name unsloth/Phi-4 --size 300 --n_digits 3
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 11 --model_name outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 --size 300 --n_digits 4
rm output.txt
python eval.py --base 11 --model_name unsloth/Phi-4 --size 300 --n_digits 4
rm output.txt
