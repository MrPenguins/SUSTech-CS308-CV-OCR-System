import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default="stage2",
                                 help='the name of your model')
        self.parser.add_argument('--epoch', type=int, default=10,
                                 help='the number of epochs')
        self.parser.add_argument('--batch_size', type=int, default=128, help='the size of the batch_size')
        self.parser.add_argument('--lr', type=int, default=1e-4, help='the learning rate')
        self.parser.add_argument('--model', type=str, default="stage2", help='choose the model you want to use')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        return self.opt


