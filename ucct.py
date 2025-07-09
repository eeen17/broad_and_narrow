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

def algorithm_2(layer_idx, T, A, alpha=1.0, beta=1.0, gamma=1.0, model_path="unsloth/Phi-4"):

    ## using the model's hidden states 
    
    embeddings = extract_embeddings(model, tokenizer, texts, layer_idx=layer_idx)
    rho_d = 1 / np.mean(pdist(embeddings, metric='cosine'))

    t = extract_embeddings(model, tokenizer, [T], layer_idx)[0] 
    p = np.mean(embeddings, axis=0)
    d_r = cosine(t, p)

    k = len(A)
    S = alpha * rho_d - beta * d_r - gamma * np.log(k)

    return S, rho_d, d_r


def extract_embeddings(model, tokenizer, texts, layer_idx=-1):

    embeddings = []
    
    for text in texts:
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                if layer_idx < 0:
                    layer_idx = len(hidden_states) + layer_idx
                
                if layer_idx >= len(hidden_states):
                    print(len(hidden_states))
                    print(f"Warning: layer_idx {layer_idx} is out of bounds. Using last layer.")
                    layer_idx = len(hidden_states) - 1
                
                # mean pooling over the seq length
                attention_mask = inputs['attention_mask']
                hidden_state = hidden_states[layer_idx]  # Shape: (batch_size, seq_len, hidden_size)
                
                masked_hidden = hidden_state * attention_mask.unsqueeze(-1)
                
                # mean pooling over seq length
                embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                embedding = embedding.float().cpu().numpy()
                embeddings.append(embedding)
                
        except Exception as e:
            print(f"error")
            # Return zero embedding as fallback
            if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                zero_embedding = np.zeros((1, model.config.hidden_size))
            else:
                zero_embedding = np.zeros((1, 4096))  # Default for Phi-4
            embeddings.append(zero_embedding)
    
    return np.vstack(embeddings)
    
def main():
    method = 1
    layer_idx = -1
    
    for base in [8, 9, 10]:
        form = "icl_cot_False" # "icl_cot_False" "icl_cot_True" "eqn_only"  
        k = 8
        
        test_data = load_data(f'arithmetic/data/0shot/base{base}.txt', 250)
        shot_data = load_data(f'ft_data/data_ft_{base}_2.txt', k)
        if form == "icl_cot_True":
            A = [f"{templatize(shot, base)} {answer(shot, base)}" for shot in shot_data]
        elif form == "icl_cot_False":
            A = [f"{templatize(shot, base)} \\boxed{{{get_label(shot, base)}}}" for shot in shot_data]
        elif form == "eqn_only":
            A = [f"{shot}={get_label(expr, base)}" for shot in shot_data]
    
        S_mean = 0
        for i in range(250):
            
            expr = test_data[i]
            T = templatize(expr, base)
            
            if method == 1:
                S, rho_d, d_r = algorithm_1(T, A)
            if method == 2:
                S, rho_d, d_r = algorithm_2(layer_idx, T, A)
                
            # print(S)
    
            S_mean += S
        print(base, S_mean / 250, rho_d)
    
        
    
main()  
    

