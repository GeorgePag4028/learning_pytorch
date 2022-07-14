#!/usr/bin/env python3
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.transform(test)

train = torch.from_numpy(train.astype(np.float32))
test = torch.from_numpy(test.astype(np.float32))
train_labels = torch.from_numpy(train_labels.astype(np.float32))
test_labels = torch.from_numpy(test_labels.astype(np.float32))

train_labels = train_labels.view(train_labels.shape[0], 1)
test_labels = test_labels.view(test_labels.shape[0], 1)

# 1) Model
# Linear model f = wx + b , sigmoid at the end
class Model(nn.Module):
	def __init__(self, n_input_features):
		super(Model, self).__init__()
		self.linear1 = nn.Linear(n_input_features, 10)
		self.linear2 = nn.Linear(10, 1)

	def forward(self, x):
		x = self.linear1(x)
		x = torch.sigmoid(self.linear2(x))
		return x

model = Model(n_features)

# 2) Loss and optimizer
num_epochs = 200
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in (t := tqdm(range(num_epochs))):
	# Forward pass and loss
	y_pred = model(train)
	loss = criterion(y_pred, train_labels)

	# Backward pass and update
	loss.backward()
	optimizer.step()

	# zero grad before new step
	optimizer.zero_grad()

	t.set_description(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
	time.sleep(0.05)


with torch.no_grad():
	y_predicted = model(test)
	y_predicted_cls = y_predicted.round()
	acc = y_predicted_cls.eq(test_labels).sum() / float(test_labels.shape[0])
	print(f'accuracy: {acc.item():.4f}')
