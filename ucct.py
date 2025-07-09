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

def algorithm_2(T, A, alpha=1.0, beta=1.0, gamma=1.0, model_path="unsloth/Phi-4"):

    ## using the model's hidden states 

    
def main():
    
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
            
            
            S, rho_d, d_r = algorithm_1(T, A)
            # print(S)
    
            S_mean += S
        print(base, S_mean / 250, rho_d)
    
        
    
main()  
    

