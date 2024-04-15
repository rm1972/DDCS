import numpy as np
import torch

mean_path = "ID_prototype/mean.npy"

class_means_dict = np.load(mean_path, allow_pickle=True).item()  # 加载为字典

class_means_list = [class_means_dict[label] for label in sorted(class_means_dict.keys())] 

class_means_tensor = torch.tensor(class_means_list, dtype=torch.float32) 

channel_variances = torch.var(class_means_tensor, dim=0) ##2048维

print("Class-wise channel variances:", channel_variances)
print(channel_variances.shape)

save_file = "var.pt" 
torch.save(channel_variances,save_file)
