#!/usr/bin/env python3
from tqdm import tqdm
import time
import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


"""for i in (t := tqdm(range(100))):
  t.set_description(f'nice:{i}')
  time.sleep(0.5)"""




def print_missing(np_data):
  rows, col = np_data.shape
  nice = np.zeros((1,16))

  for i in range(rows):
    missing_line = False
    for j in range(col-1):
      if np_data[i][j] == '?':
        missing_line = True
    if missing_line == False:
      nice = np.append(nice,[np_data[i,:]],axis=0)
  return nice

def label(np_data):
  return np_data[:,0:15], np_data[:,15]

def class_distribution(np_data):
  print(f"+:  {np.sum(np_data == '+')} , {np.sum(np_data == '+')/np_data.shape[0]} %")
  print(f"-:  {np.sum(np_data == '-')} , {np.sum(np_data == '-')/np_data.shape[0]} %")

class TinyModel(torch.nn.Module):
  def __init__(self):
    super(TinyModel, self).__init__()
    
    self.linear1 = nn.Linear(16,10)
    self.relu1 = nn.ReLU()
    self.linear2 = nn.Linear(10,2)
    self.relu2 = nn.ReLU()

  def forward(self,x):
    x = self.relu1(self.linear1(x))
    x = self.relu2(self.linear2(x))
    return x


if __name__ == '__main__':
  data = pd.read_csv('uci_data/crx.data',header = None)
  
  data = data.values
  print(data.shape)
  data = print_missing(data)
  print(data.shape)
  
  nice, labels = label(data)
  print(nice.shape)
  print(labels.shape)

  class_distribution(labels)

  train_dataset = data[: int(0.8 * len(data)),:]
  test_dataset = data[int(0.8 * len(data)):,:]

  train, train_labels = label(train_dataset)
  test, test_labels = label(test_dataset)
  print(train.shape)
  print(train_labels.shape)
  print(test.shape)
  print(test_labels.shape)

  net = TinyModel()
  for epoch in range(2):
    outputs = net(torch.from_numpy(train))

  print("end")






