# %%
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download, notebook_login
import numpy as np
import torch

# %%
#notebook_login()

# %%
filenames = [
    "layer_12/width_16k/average_l0_22/params.npz",
    "layer_12/width_32k/average_l0_22/params.npz",
    "layer_12/width_65k/average_l0_21/params.npz",
    "layer_12/width_262k/average_l0_21/params.npz",
    "layer_12/width_524k/average_l0_22/params.npz",
    "layer_12/width_1m/average_l0_19/params.npz",
    "layer_20/width_16k/average_l0_71/params.npz",
]
target_layers = [12, 12, 12, 12, 12, 12, 20]
target_widths = ["16k", "32k", "65k", "262k", "524k", "1m", "16k"]
assert len(filenames) == len(target_layers)

# %%

models_to_loads = [[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3]] # [0,1,2,3,4,5]
for models_to_load in models_to_loads:
    print("Starting with models", models_to_load)
    path_to_params_allmodel = []
    target_layer_allmodel = []
    for i in models_to_load:
        path_to_params_allmodel.append(hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-res",
            filename=filenames[i],
            force_download=False,
        ))
        target_layer_allmodel.append(target_layers[i])

    # %%
    params_allmodel = [np.load(path_to_params) for path_to_params in path_to_params_allmodel]
    #pt_params_allmodel = [{k: torch.from_numpy(v).cuda() for k, v in params.items()} for params in params_allmodel]
    pt_params_allmodel = [{k: torch.from_numpy(v) for k, v in params.items()} for params in params_allmodel]

    # %%
    from itertools import combinations
    from tqdm import tqdm
    import random
    from collections import defaultdict

    # sanity check whether all vectors are unit length
    for pt_params in pt_params_allmodel:
        len_of_vectors = pt_params['W_dec'].square().sum(-1)
        assert torch.allclose(len_of_vectors, torch.tensor(1.0))

    # Since there are too many vector pairs, we will choose only a random sample of them
    NUM_OF_VECTORPAIRS = 1500
    pt_params_small_model = pt_params_allmodel[0]
    pt_params_large_model = pt_params_allmodel[1]
    num_vectors_small = pt_params_small_model['W_dec'].size(0)
    num_vectors_large = pt_params_large_model['W_dec'].size(0)

    vector_idxs_large = random.sample(range(num_vectors_large), NUM_OF_VECTORPAIRS)
    vector_idxs_large_dict = {vector_idxs_large[i]:i for i in range(len(vector_idxs_large))}

    # vector_idxs_large = range(num_vectors_large) # Does not work: requies too much memory
    vector_pairs = [(i, j) for j in vector_idxs_large for i in range(num_vectors_small)]
    #cos_sims = defaultdict(lambda: {"head": None, "value": -1})
    # Cosinus similarity
    cos_sim_matrix = np.array([[0 for j in range(len(vector_idxs_large))] for i in range(num_vectors_small)], dtype="float16")
    for (i,j) in tqdm(vector_pairs):
        cos_sim = torch.dot(pt_params_small_model['W_dec'][i],pt_params_large_model['W_dec'][j]).item()
        cos_sim_matrix[i,vector_idxs_large_dict[j]] = cos_sim

    # %%
    # Hist graph about the cos sim
    import matplotlib.pyplot as plt
    max_values = [cos_sim_matrix[:,i].max() for i in range(cos_sim_matrix.shape[1])]
    fig, ax = plt.subplots()
    ax.hist(max_values, bins=50)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("cosine Similarity")
    ax.set_title(f"Maximum Cosine Similarity of Large SAE ({target_widths[models_to_load[1]]}) Features\n with Small SAE ({target_widths[models_to_load[0]]}) Features")
    # write mean, median and stdev on a legend
    mean = np.mean(max_values)
    median = np.median(max_values)
    stdev = np.std(max_values)
    ax.legend([f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStdev: {stdev:.2f}"])
    plt.savefig(f"cos_sim_hist_{models_to_load[0]}_{models_to_load[1]}.png")

    # %%
    node_degree = [sum(cos_sim_matrix[:,i] > 0.7) for i in range(cos_sim_matrix.shape[1])]
    # group by numbers of node degree
    from collections import defaultdict
    degrees = defaultdict(int)
    for elem in node_degree:
        degrees[elem] += 1
    print(degrees)
    # save degrees to file
    with open(f"degrees.txt", "a") as f:
        f.write(f"Model {models_to_load[0]} and {models_to_load[1]}\n")
        for key, value in degrees.items():
            f.write(f"{key}:{value}, ")
        f.write("\n")
