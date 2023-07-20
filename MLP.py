import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

#arg parse
parser = argparse.ArgumentParser(description='MLP')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--layer_num', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch_norm', type=bool, default=False)

args = parser.parse_args()
