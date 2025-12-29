import torch
import os

input_path = r"D:\Ope Watson\Project2\XTTSv2-Finetuning-for-New-Languages\OldModels\dvaethiene32gptcluster1\best_model_20570.pth"  # File 5GB
output_path = r"D:\Ope Watson\Project2\XTTSv2-Finetuning-for-New-Languages\OldModels\dvaethiene32gptcluster1\best_model_20570_pruned.pth"

checkpoint = torch.load(input_path, map_location="cpu")
print("Keys gốc:", list(checkpoint.keys()))

if 'optimizer' in checkpoint:
    del checkpoint['optimizer']
    print("Đã xóa optimizer")

if 'model' in checkpoint:
    for key in list(checkpoint['model'].keys()):
        if 'dvae' in key.lower():
            del checkpoint['model'][key]
    print("Đã xóa DVAE keys")

torch.save(checkpoint, output_path)
print(f"Giảm từ {os.path.getsize(input_path)/1e9:.1f}GB → {os.path.getsize(output_path)/1e9:.1f}GB")
