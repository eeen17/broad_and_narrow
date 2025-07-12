import numpy as np
import torch
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist, cosine

from arithmetic.eval import get_label
from arithmetic.query import load_data, templatize, answer
from infer import load_model


def algorithm_1(T, A, alpha=1.0, beta=1.0, gamma=1.0):

    ## external embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(A)
    rho_d = 1 / np.mean(pdist(embeddings, metric='cosine'))

    t = model.encode([T])[0]
    p = np.mean(embeddings, axis=0)
    d_r = cosine(t, p)

    k = len(A)
    S = alpha * rho_d - beta * d_r - gamma * np.log(k)

    return S, rho_d, d_r

def algorithm_2(model, tokenizer, layer_idx, T, A, alpha=1.0, beta=1.0, gamma=1.0, pooling_method="mean"):

    embeddings = extract_embeddings(model, tokenizer, A, layer_idx=layer_idx, pooling_method=pooling_method)
    rho_d = 1 / np.mean(pdist(embeddings, metric='cosine'))

    t = extract_embeddings(model, tokenizer, [T], layer_idx, pooling_method=pooling_method)[0]
    p = np.mean(embeddings, axis=0)
    d_r = cosine(t, p)

    k = len(A)
    S = alpha * rho_d - beta * d_r - gamma * np.log(k)

    return S, rho_d, d_r


def extract_embeddings(model, tokenizer, texts, layer_idx=-1, pooling_method="mean"):

    embeddings = []

    for text in texts:
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                if layer_idx < 0:
                    layer_idx = len(hidden_states) + layer_idx

                if layer_idx >= len(hidden_states):
                    layer_idx = len(hidden_states) - 1

                hidden_state = hidden_states[layer_idx]  # Shape: (batch_size, seq_len, hidden_size)

                if pooling_method == "mean":
                    attention_mask = inputs['attention_mask']
                    masked_hidden = hidden_state * attention_mask.unsqueeze(-1)
                    # mean pooling over seq length
                    embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                elif pooling_method == "final_tok":
                    embedding = hidden_state[:, -1, :]
                elif pooling_method == "eos":
                    attention_mask = inputs['attention_mask']
                    eos_positions = attention_mask.sum(dim=1) - 1
                    embedding = hidden_state[torch.arange(hidden_state.size(0)), eos_positions, :]
            


                embedding = embedding.float().cpu().numpy()
                embeddings.append(embedding)

        except Exception as e:
            print(f"error: {e}")
            if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                zero_embedding = np.zeros((1, model.config.hidden_size))
            else:
                zero_embedding = np.zeros((1, 4096))  # Default for Phi-4
            embeddings.append(zero_embedding)

    return np.vstack(embeddings)

def main():
    method = 2
    layer_idx = -1
    pooling_method = "eos" # "mean", "final_tok", "eos"
    ## using the model's hidden states
    model, tokenizer = FastLanguageModel.from_pretrained(
                  model_name="unsloth/Phi-4",
                  max_seq_length=8,
                  load_in_4bit=True,
              )

    for base in [8, 9, 10]:
        form = "eqn_only" # "icl_cot_False" "icl_cot_True" "eqn_only"
        k = 8

        test_data = load_data(f'arithmetic/data/0shot/base{base}.txt', 250)
        shot_data = load_data(f'ft_data/data_ft_{base}_2.txt', k)
        if form == "icl_cot_True":
            A = [f"{templatize(shot, base)} {answer(shot, base)}" for shot in shot_data]
        elif form == "icl_cot_False":
            A = [f"{templatize(shot, base)} \\boxed{{{get_label(shot, base)}}}" for shot in shot_data]
        elif form == "eqn_only":
            A = [f"{expr}={get_label(expr, base)}" for expr in shot_data]

        S_mean = 0
        for i in range(250):

            expr = test_data[i]
            T = templatize(expr, base)

            if method == 1:
                S, rho_d, d_r = algorithm_1(T, A)
            if method == 2:

              S, rho_d, d_r = algorithm_2(model, tokenizer, layer_idx, T, A, pooling_method=pooling_method)

            # print(S)

            S_mean += S
        print(base, S_mean / 250, rho_d)
