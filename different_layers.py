# %%
from huggingface_hub import hf_hub_download, notebook_login
import numpy as np
import torch
from itertools import combinations

# %%
#notebook_login()

# %%
filenames = [
    "layer_0/width_16k/average_l0_25/params.npz",
    "layer_1/width_16k/average_l0_20/params.npz",
    "layer_2/width_16k/average_l0_24/params.npz",
    # "layer_3/width_16k/average_l0_28/params.npz",
    "layer_10/width_16k/average_l0_21/params.npz",
    "layer_11/width_16k/average_l0_22/params.npz",
    "layer_12/width_16k/average_l0_22/params.npz",
    # "layer_13/width_16k/average_l0_23/params.npz",
    # "layer_20/width_16k/average_l0_22/params.npz",
    "layer_21/width_16k/average_l0_22/params.npz",
    "layer_22/width_16k/average_l0_21/params.npz",
    "layer_23/width_16k/average_l0_21/params.npz",
]
target_layers = [0,1,2,10,11,12,21,22,23]
# target_layers = [0,1,3,10,13,21,22]
target_widths = ["16k"] * len(target_layers)
assert len(filenames) == len(target_layers)

# %%
res = np.zeros((len(filenames), len(filenames)), dtype=float)
models_to_loads = combinations(range(len(filenames)), 2)
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

    params_allmodel = [np.load(path_to_params) for path_to_params in path_to_params_allmodel]
    #pt_params_allmodel = [{k: torch.from_numpy(v).cuda() for k, v in params.items()} for params in params_allmodel]
    pt_params_allmodel = [{k: torch.from_numpy(v) for k, v in params.items()} for params in params_allmodel]

    # %%
    from tqdm import tqdm
    import random
    from collections import defaultdict

    # sanity check whether all vectors are unit length
    for pt_params in pt_params_allmodel:
        len_of_vectors = pt_params['W_dec'].square().sum(-1)
        assert torch.allclose(len_of_vectors, torch.tensor(1.0))

    # Since there are too many vector pairs, we will choose only a random sample of them
    NUM_OF_VECTORPAIRS = 50
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
    ax.set_ylabel("Frequency")
    ax.set_xlabel("cosine Similarity")
    ax.set_title(f"Maximum Cosine Similarity of layer ({target_layers[models_to_load[0]]}) Features\n with layer  ({target_layers[models_to_load[1]]})")
    # write mean, median and stdev on a legend
    mean = np.mean(max_values)
    median = np.median(max_values)
    stdev = np.std(max_values)
    ax.legend([f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStdev: {stdev:.2f}"])
    plt.savefig(f"generated_figs/cos_sim_hist_{models_to_load[0]}_{models_to_load[1]}.png")
    proportion = sum(1 for i in max_values if i > 0.7) / len(max_values)
    print(f"{proportion*100}% vectors have cosine similarity > 0.7")
    res[models_to_load[0], models_to_load[1]] = proportion
print(res)
with open("cos_sim_res.txt", "w") as f:
    f.write(str(res))
# make a heatmap
import seaborn as sns
import pandas as pd

# Mask to keep only the upper triangle
mask = np.tril(np.zeros_like(res, dtype=bool))

# Create a DataFrame
df = pd.DataFrame(res, columns=target_layers, index=target_layers)

plt.figure(figsize=(10, 8))
# Create the heatmap with the mask
sns.heatmap(df, annot=True, mask=mask, cmap="viridis")
plt.xlabel("Layer")
plt.ylabel("Layer")
plt.savefig("generated_figs/cos_sim_heatmap.png")
# Create the heatmap without masking