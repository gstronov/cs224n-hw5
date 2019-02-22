#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
### YOUR CODE HERE for part 1h

import torch.nn.functional
import torch
import torch.nn as nn

class Highway(nn.Module):
    def __init__(self, e_word):
        super(Highway, self).__init__()
        self.W_proj = nn.Linear(e_word, e_word)
        self.W_gate = nn.Linear(e_word, e_word)

    def forward(self, x_conv_out):
        x_proj = torch.nn.functional.relu(self.W_proj(x_conv_out))
        x_gate = torch.nn.functional.sigmoid(self.W_gate(x_conv_out))
        x_highway = torch.mul(x_gate, x_proj) + torch.mul((1 - x_gate), x_conv_out)
        return x_highway
### END YOUR CODE 

