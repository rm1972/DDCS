
import torch
import os
import argparse

sim_path = "similarity.pt"

S = torch.load(sim_path)  

var_path = "var_result.pt"

V = torch.load(var_path) 

parser = argparse.ArgumentParser(description='Description of your program')

parser.add_argument('--a', type=float, help='Description of parameter a')

args = parser.parse_args()

a = args.a

print("Value of a:", a)

J=a*S-(1-a)*V

sorted_indices = torch.argsort(J, descending=True)

output_dir = "select_k/a={}".format(a)
os.makedirs(output_dir, exist_ok=True)


for K in range(1281):
   
    selected_channels = sorted_indices[-K:]

    filename = f"selected_channels_{K}.pt"

    filepath = os.path.join(output_dir, filename)
    torch.save(selected_channels, filepath)
  
    print(f"Saved selected channels for K = {K} to {filepath}")



