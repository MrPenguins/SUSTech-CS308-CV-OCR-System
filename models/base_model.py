import os
import torch
import sys


class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        pass

    def forward(self,input):
        pass