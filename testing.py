import torch
ckpt = torch.load('work_dirs/openlane/small/epoch_2.pth', map_location='cpu')
print(ckpt['meta']['config'])