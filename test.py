#!/usr/bin/env python3
import torch 
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

x = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8], dtype=np.float32)

w =0.0

def forward(x):
  return w*x

def loss(y,y_pred):
  return ((y_pred -y) **2).mean()

def gradient(x, y, y_pred):
  return np.dot(2*x, y_pred-y).mean()

print(f'Prediction before training: f(5)= {forward(5):.3f}')

learning_rate = 0.01
n_iters = 20

for epoch in (t:= tqdm(range(n_iters))):
  y_pred = forward(x)
  l = loss(y, y_pred)
  dw = gradient(x, y, y_pred)

  w -= learning_rate * dw
  print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
  
print(f'Prediction before training: f(5)= {forward(5):.3f}')
