# follow https://wikidocs.net/60314
import urllib.request
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random
import math
import time
import dill

class MeanPooler(nn.Module):
    def __init__(self, hid_dim ,dropout_rate):
        super().__init__()
        self.dense1 = nn.Linear(hid_dim, hid_dim)
        self.dense2 = nn.Linear(hid_dim, hid_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        
    def forward(self, hidden_states, mask, sqrt=True):
        # hidden states: [N, batch_size, seq, model_dim]
        # attention masks: [N, batch_size, seq]
       
        hidden_states = hidden_states[:,:mask.size(2)]
        sentence_sums = torch.matmul(mask.float() , hidden_states) # B x N x dim
        divisor = mask.sum(dim=2).unsqueeze(-1) # B x N x 1


        if sqrt :
            divisor = torch.clamp(divisor,min=1)
            divisor = divisor.sqrt()
            sentence_sums /= divisor
        
        pooled_output = self.dense1(sentence_sums)
        pooled_output = self.relu(pooled_output)
        
        pooled_output = self.dropout(pooled_output)
        
        pooled_output = self.dense2(pooled_output)
        

        return pooled_output # B x N x dim
        
'''
class MeanPooler(nn.Module):
    def __init__(self,hidden_dim,dropout_rate):
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.Relu   = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, mask, sqrt=False):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]

        sentence_sums = torch.bmm(hidden_states.permute(0, 2, 1), mask.float().unsqueeze(-1)).squeeze(-1)
        
        divisor = mask.sum(dim=1).view(-1, 1).float()
        if sqrt:
            divisor = divisor.sqrt()

        sentence_sums /= divisor

        pooled_output = self.dense1(sentence_sums)
        pooled_output = self.Relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.dense2(pooled_output)

        return pooled_output
'''
