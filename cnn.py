#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn.functional
import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, filters, char_embed, kernel=5):
        super(CNN, self).__init__()
        self.W_cnn = nn.Convid(char_embed, filters, kernel_size=kernel, stride=1)
        self.max_pool = nn.MaxPoolid(kernel_size=21)

    def forward(self, x_reshaped):
        x_conv = self.W_cnn(x_reshaped)
        #x_convout = torch.nn.functional.relu(torch.max(x_conv, dim=2)[0])
        x_convout = torch.nn.functional.relu(self.max_pool(x_conv).squeeze(2))
        return x_convout
### END YOUR CODE

