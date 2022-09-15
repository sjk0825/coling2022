import numpy as np
import torch

def getNpy(batch, step, node_idx, node_len, node_edge, data_size, seq_len, device):
    
    start = batch*step
    
    end   = start + data_size
    seq_len = min(seq_len, 200)
    node_idx = node_idx[start:end, :50, :seq_len]
    node_len = node_len[start:end]
    node_edge = node_edge[start:end, :50,:50]

    node_idx = torch.from_numpy(node_idx).to(device)
    node_len = torch.from_numpy(node_len).to(device)
    node_edge = torch.from_numpy(node_edge).to(device)


    return node_idx, node_len, node_edge

    
