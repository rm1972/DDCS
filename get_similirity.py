
import numpy as np
import torch

mean_path = "ID_prototype/mean.npy"

class_means_dict = np.load(mean_path, allow_pickle=True).item()  

class_means_list = [class_means_dict[label] for label in sorted(class_means_dict.keys())] 

class_means_tensor = torch.tensor(class_means_list, dtype=torch.float32)

cate_num=1000
feat_dim=1280
save_file = "similarity.pt"

mean = class_means_tensor.unsqueeze(1)
feats = mean.squeeze()
sim_sum = torch.zeros((feat_dim))
count = 0
for i in range(cate_num):
    for j in range(cate_num):
        if i != j:
            sim_sum += feats[i, :] * feats[j, :]
            count += 1
sim = sim_sum / count
torch.save(sim, save_file)



