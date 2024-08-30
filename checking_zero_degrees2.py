# %%
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download, notebook_login
import numpy as np
import torch
from tqdm import tqdm
import random
from collections import defaultdict


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
    "layer_12/width_16k/average_l0_82/params.npz",
    "layer_12/width_32k/average_l0_76/params.npz",
    "layer_20/width_16k/average_l0_71/params.npz",
]
target_widths = ["16k", "32k", "65k", "262k", "524k", "1m", "16k", "32k", "16k"]
target_layers = [12, 12, 12, 12, 12, 12, 12, 12, 20]
assert len(filenames) == len(target_layers) and len(filenames) == len(target_widths)

# %%
def get_cos_sim_matrix(models_to_load, NUM_OF_VECTORPAIRS = 1500, vector_idxs_large = None):
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

	# sanity check whether all vectors are unit length
	for pt_params in pt_params_allmodel:
		len_of_vectors = pt_params['W_dec'].square().sum(-1)
		assert torch.allclose(len_of_vectors, torch.tensor(1.0))

	# Since there are too many vector pairs, we will choose only a random sample of them
	pt_params_small_model = pt_params_allmodel[0]
	pt_params_large_model = pt_params_allmodel[1]
	num_vectors_small = pt_params_small_model['W_dec'].size(0)
	num_vectors_large = pt_params_large_model['W_dec'].size(0)

	if vector_idxs_large is None:
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
	return cos_sim_matrix, vector_idxs_large

# %%
cos_sim_matrix, vector_idxs_large  = get_cos_sim_matrix([1, 0])

# %%
node_degree = [sum(cos_sim_matrix[:,i] > 0.7) for i in range(cos_sim_matrix.shape[1])]
zero_degrees = [vector_idxs_large[i] for i in range(len(node_degree)) if node_degree[i] == 0]

# %%
cos_sim_matrix_2, vector_idxs_large_2  = get_cos_sim_matrix([2, 0], vector_idxs_large=zero_degrees)
node_degree_2 = [sum(cos_sim_matrix_2[:,i] > 0.7) for i in range(cos_sim_matrix_2.shape[1])]
zero_degrees_2 = [vector_idxs_large_2[i] for i in range(len(node_degree_2)) if node_degree_2[i] == 0]
print(len(zero_degrees_2))
# %%
cos_sim_matrix_3, vector_idxs_large_3  = get_cos_sim_matrix([3, 0], vector_idxs_large=zero_degrees_2)
node_degree_3 = [sum(cos_sim_matrix_3[:,i] > 0.7) for i in range(cos_sim_matrix_3.shape[1])]
zero_degrees_3 = [vector_idxs_large_3[i] for i in range(len(node_degree_3)) if node_degree_3[i] == 0]
print(len(zero_degrees_3))
# %%
cos_sim_matrix_4, vector_idxs_large_4  = get_cos_sim_matrix([4, 0], vector_idxs_large=zero_degrees_3)
node_degree_4 = [sum(cos_sim_matrix_4[:,i] > 0.7) for i in range(cos_sim_matrix_4.shape[1])]
zero_degrees_4 = [vector_idxs_large_4[i] for i in range(len(node_degree_4)) if node_degree_4[i] == 0]
print(len(zero_degrees_4))

# %%
with open("zero_degrees_4.txt", "w") as f:
	for item in zero_degrees_4:
		f.write("%s\n" % item)
# %%
html_template = "https://neuronpedia.org/{}/{}/{}"

def get_dashboard_html(sae_release = "gemma-2-2b", sae_id="20-gemmascope-res-16k", feature_idx=0):
    return html_template.format(sae_release, sae_id, feature_idx)

with open("zero_degrees_4_html.txt", "w") as f:
	for item in zero_degrees_4:
		html = get_dashboard_html(sae_release = "gemma-2-2b", sae_id="12-gemmascope-res-16k", feature_idx=item)
		f.write("%s\n" % html)


# %%
print("Number of zero degrees:", len(zero_degrees), len(zero_degrees_2), len(zero_degrees_3), len(zero_degrees_4))
max_values_zero_degree = [cos_sim_matrix[:,i].max() for i in range(10) if node_degree[i] == 0]
max_values_zero_degree_2 = [cos_sim_matrix_2[:,i].max() for i in range(10) if node_degree_2[i] == 0]
max_values_zero_degree_3 = [cos_sim_matrix_3[:,i].max() for i in range(10) if node_degree_3[i] == 0]
max_values_zero_degree_4 = [cos_sim_matrix_4[:,i].max() for i in range(10) if node_degree_4[i] == 0]
print("Max values of zero degrees:")
print(sum(max_values_zero_degree) / len(max_values_zero_degree))
print(sum(max_values_zero_degree_2) / len(max_values_zero_degree_2))
print(sum(max_values_zero_degree_3) / len(max_values_zero_degree_3))
print(sum(max_values_zero_degree_4) / len(max_values_zero_degree_4))
# %%
