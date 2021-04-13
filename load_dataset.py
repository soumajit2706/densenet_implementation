import pandas as pd 
import numpy as np 
import torch
from torch.utils.data import Dataset,DataLoader


train_dataset = pd.read_csv('yahoo_answers_csv/train.csv',header=None)
test_dataset = pd.read_csv('yahoo_answers_csv/test.csv',header=None)


train_input_tensor = torch.zeros(train_dataset.shape[0], 1024)
train_label_tensor = torch.zeros(train_dataset.shape[0],dtype=torch.long)
test_input_tensor = torch.zeros(test_dataset.shape[0], 1024)
test_label_tensor = torch.zeros(test_dataset.shape[0],dtype=torch.long)

for j in range(train_dataset.shape[0]):
    if len(str(train_dataset.loc[j,3])) < 1024: 
        train_input_tensor[j,:len(str(train_dataset.loc[j,3]))] = torch.FloatTensor([float(ord(i)) for i in list(str(train_dataset.loc[j,3]))])
    
    else: 
        train_input_tensor[j,:1024] = torch.FloatTensor([float(ord(i)) for i in list(str(train_dataset.loc[j,3][:1024]))])
    
    train_label_tensor[j] = int(train_dataset.loc[j,0]-1)


for j in range(test_dataset.shape[0]):
    if len(str(test_dataset.loc[j,3])) < 1024:
        test_input_tensor[j,:len(str(test_dataset.loc[j,3]))] = torch.FloatTensor([float(ord(i)) for i in list(str(test_dataset.loc[j,3]))])

    else:
        test_input_tensor[j,:1024] = torch.FloatTensor([float(ord(i)) for i in list(str(test_dataset.loc[j,3][:1024]))])

    test_label_tensor[j] = int(test_dataset.loc[j,0]-1)

torch.save(train_input_tensor,'./yahoo_answers_csv/train_input_tensor.pt')
torch.save(train_label_tensor,'./yahoo_answers_csv/train_label_tensor.pt')
torch.save(test_input_tensor,'./yahoo_answers_csv/test_input_tensor.pt')
torch.save(test_label_tensor,'./yahoo_answers_csv/test_label_tensor.pt')

