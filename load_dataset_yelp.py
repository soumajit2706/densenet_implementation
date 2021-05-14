import pandas as pd 
import numpy as np 
import torch
from torch.utils.data import Dataset,DataLoader
from sys import exit

####################################### load csv files #####################################
train_dataset = pd.read_csv('yelp_review_polarity_csv/train.csv',header=None)
test_dataset = pd.read_csv('yelp_review_polarity_csv/test.csv',header=None)

####################################### initialization #####################################
train_input_tensor = torch.zeros(train_dataset.shape[0], 1024)
train_label_tensor = torch.zeros(train_dataset.shape[0],dtype=torch.long)
test_input_tensor = torch.zeros(test_dataset.shape[0], 1024)
test_label_tensor = torch.zeros(test_dataset.shape[0],dtype=torch.long)
chrs = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\\/|_@#$%^&*~`+-=<>()[]{} "
#chrs = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\\/|_@#$%^&*~`+-=<>()[]{}\n "
lookup_table = {}
for i, c in enumerate(chrs):
    print(c,i)
    lookup_table[c] = i+1
    
####################################### creating the training and test data #####################################
for j in range(train_dataset.shape[0]):
    flag = str(train_dataset.loc[j,1]) #+ str(train_dataset.loc[j,2])+str(train_dataset.loc[j,3])
    char = list(flag)
    for i,c in enumerate(char):
        if c not in lookup_table.keys():
            train_input_tensor[j,i] = 0
        else:
            train_input_tensor[j,i] = lookup_table[c]
        if i == 1023:
            break
    train_label_tensor[j] = int(train_dataset.loc[j,0]-1)
    del flag
    del char
for j in range(test_dataset.shape[0]):
    flag = str(test_dataset.loc[j,1]) #+ str(test_dataset.loc[j,2]) +str(test_dataset.loc[j,3])
    char = list(flag)
    for i,c in enumerate(char):
        if c not in lookup_table.keys():
            test_input_tensor[j,i] = 0
        else:
            test_input_tensor[j,i] = lookup_table[c]
        if i == 1023:
            break
    test_label_tensor[j] = int(test_dataset.loc[j,0]-1)
    del flag
    del char

####################################### saving the training and test data #####################################
torch.save(train_input_tensor,'./yelp_review_polarity_csv/train_input_tensor.pt')
torch.save(train_label_tensor,'./yelp_review_polarity_csv/train_label_tensor.pt')
torch.save(test_input_tensor,'./yelp_review_polarity_csv/test_input_tensor.pt')
torch.save(test_label_tensor,'./yelp_review_polarity_csv/test_label_tensor.pt')
del train_input_tensor
del test_input_tensor
