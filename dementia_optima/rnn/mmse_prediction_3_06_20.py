#!/usr/bin/env python
# coding: utf-8

# # **Dementia Patients -- Analysis and Prediction**
# ### ***Author : Akhilesh Vyas***
# ### ****Date : May, 2020****

# - <a href='#00'>0. Setup </a>
#     - <a href='#00.1'>0.1. Load libraries </a>
#     - <a href='#00.2'>0.2. Define paths </a>
# 
# - <a href='#01'>1. Data Cleaning and Preprocessing </a>  
#     
# - <a href='#02'>2. Deep Neural Network Model</a>
# 
# - <a href='#03'>3. Result Analysis</a> 

# # <a id='00'>0. Setup </a>

# ## <a id='#00.1'>0.1. Load libraries </a>

# In[ ]:


import torch
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.model_selection import train_test_split


# ## <a id='#00.2'>0.2. Define paths </a>

# In[ ]:


data_path = '../../data/'


# # <a id='#01'>1. Data Cleaning and Preprocessing </a>

# In[ ]:


patient_df = pd.read_csv(data_path+'patient_df_rnn.csv')

#print (patient_df.isna().sum()/patient_df.shape[0])
patient_df = patient_df.loc[:, patient_df.isna().sum()/patient_df.shape[0]< 0.4]

patient_df['patient_index'] = patient_df.index
patient_ids = patient_df['patient_id']
patient_eps = patient_df['EPISODE_DATE']

patient_df = patient_df.drop(columns=['EPISODE_DATE', 'MINI_MENTAL_SCORE_PRE'])
patient_df = patient_df.groupby(by='patient_id').transform(lambda x: x.interpolate(method='ffill'))

patient_df['EPISODE_DATE'] = patient_eps
patient_df['patient_id'] = patient_ids

patient_df = patient_df.loc[:, patient_df.isna().sum()/patient_df.shape[0]< 0.2]

patient_df = patient_df.iloc[:, [-1,-2,-3]+ [i for i in range(0, len(patient_df.columns)-3)]]

patient_df = patient_df.fillna(-1)
patient_df.head(5)


# In[ ]:


# embedding Matrix
#embedding_weight = F.normalize(torch.from_numpy(patient_df.iloc[:, 4:].values).float(), p=2, dim=1, eps=1e-12, out=None)
embedding_weight = torch.from_numpy(patient_df.iloc[:, 4:].values).float()
print(embedding_weight.shape)
# pairs 
patient_indx_MMS = patient_df.groupby(['patient_id'])['patient_index', 'MINI_MENTAL_SCORE'].agg(lambda x : x.tolist())
patient_indx_MMS['count'] = patient_indx_MMS['patient_index'].apply(lambda x: len(x))


# # <a id='#02'>2. Deep Neural Network Model</a>

# In[ ]:


MAX_LENGTH = 10
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print (device)


# In[ ]:


pid_list = patient_indx_MMS['patient_index'].values
mmse_list = patient_indx_MMS['MINI_MENTAL_SCORE'].values
pairs = [[pid, mmse] if len(pid) <= 10 else [pid[0:10], mmse[0:10]] for pid, mmse  in zip(pid_list,mmse_list)]
pairs[0:10]

def tensorsFromPair(pair):
    input_tensor = torch.tensor(pair[0], dtype=torch.long, device=device).view(-1, 1)
    target_tensor = torch.tensor(pair[1], dtype=torch.float, device=device).view(-1, 1)
    return (input_tensor, target_tensor)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points1, points2):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=5)
    ax.yaxis.set_major_locator(loc)
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.plot(points1, label='train')
    plt.plot(points2, label='val')
    ax.legend()
    plt.savefig('train_val.png')


# In[ ]:


class LSTMModel(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_weight):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        # print(output.dtype, hidden.dtype)
        # output = F.relu(output) # May be not converging due to this 
        # print(output.dtype, hidden.dtype)
        output, hidden = self.lstm(output, hidden)
        #print (output)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))

'''embedding_weight = torch.Tensor(torch.randn((10,5)))

lstm_model = LSTMModel(5, 1, embedding_weight)
hidden = lstm_model.initHidden()

input_seq = torch.LongTensor([[1,2,4,5],[4,1,4,0,],[8,0,0,0]]) # max*batch
input = torch.LongTensor([[1]]) # max*batch
input_lengths = torch.Tensor([3,2,2,1])

output, hidden = lstm_model(input, hidden)
print(output[0].shape), print(hidden[0].size()) # output, h, c


print('Parameters:')
for param in lstm_model.named_parameters():
    print(type(param), param[0], param[1].size(), param[1].requires_grad)
lstm_model_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)'''

# print(lstm_model_optimizer)
# In[ ]:


def train(input_tensor, target_tensor, lstm_model, lstm_model_optimizer, criterion):
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    lstm_model_optimizer.zero_grad()
    lstm_model_hidden = lstm_model.initHidden()
    input_length = input_tensor.size(0)
    loss = 0

    for di in range(input_length):
        # print (input_tensor[di].dtype, lstm_model_hidden.dtype)
        lstm_model_output, lstm_model_hidden  = lstm_model(
            input_tensor[di], lstm_model_hidden)
        #print (lstm_model_output.dtype, target_tensor[di].dtype)
        #print (lstm_model_output, target_tensor[di])
        loss += criterion(lstm_model_output[0], target_tensor[di])
        #print (loss.requires_grad)
        #print ('train:',loss.dtype)
    
    loss.backward()
    lstm_model_optimizer.step()

    return loss.item() / input_length

def evalu(input_tensor, target_tensor, lstm_model, criterion):
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        lstm_model_hidden = lstm_model.initHidden()
        input_length = input_tensor.size(0)
        loss = 0
        output = []
        o_act = []
        o_pre = []
        for di in range(input_length):
            # print (input_tensor[di].dtype, lstm_model_hidden.dtype)
            lstm_model_output, lstm_model_hidden  = lstm_model(
                input_tensor[di], lstm_model_hidden)
            #print (lstm_model_output.dtype, target_tensor[di].dtype)
            #print (lstm_model_output, target_tensor[di])
            loss += criterion(lstm_model_output[0], target_tensor[di])
            o_pre.append(lstm_model_output[0].cpu().numpy().tolist()[0])
            o_act.append(target_tensor[di][0].cpu().numpy().tolist())

            #print (loss.requires_grad)
            #print ('train:',loss.dtype)
        output.append((o_act,o_pre))
    return (loss.item() / input_length, output)


# In[ ]:


def trainIters(lstm_model, n_iters, print_every=100, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    plot_loss_val = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    lstm_model_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
    
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    
    print (len(train_pairs), len(val_pairs))
    
    training_pairs = [tensorsFromPair(random.choice(train_pairs))
                      for i in range(n_iters)]
    
    validation_pairs = [tensorsFromPair(random.choice(val_pairs))
                      for i in range(len(val_pairs))]
    
    # need to define loss function
    criterion = nn.MSELoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, lstm_model,
                     lstm_model_optimizer, criterion)
        
        #print ('trainIters:',loss.dtype)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            
            total_val_loss = 0
            lstm_model.eval()
            output_total = []
            for i in range(0, len(validation_pairs)):
                input_tensor = validation_pairs[i][0]
                target_tensor = validation_pairs[i][1]
                loss1, output = evalu(input_tensor, target_tensor, lstm_model,criterion)
                total_val_loss+=loss1
                output_total.append(output)
            print_val_loss_avg = total_val_loss / len(validation_pairs)
            
            print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg, print_val_loss_avg))
            
            plot_loss_val.append(print_val_loss_avg)
            
            lstm_model.train()      

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
            
    return (plot_losses, plot_loss_val, output_total)




hidden_size = 171
output_size = 1

lstm_model = LSTMModel(hidden_size, output_size, embedding_weight).to(device)
#trainIters(lstm_model, 3, print_every=1)

print ('#######Started#####')

plot_losses, plot_loss_val, output_total = trainIters(lstm_model, 50000, print_every=5000, plot_every=5000)

hidden_size = 171
output_size = 1


# In[ ]:


showPlot(plot_losses, plot_loss_val)


# # <a id='#03'>3. Result Analysis</a>

# In[ ]:
import pickle

with open('output_total_sc.pkl', 'wb') as f:
    pickle.dump(output_total, f)
    
with open('output_total_sc.pkl', 'rb') as f:
    pkkk = pickle.load(f)

    def listgen(pkkk):
    lnth = len(pkkk)
    i = 0
    while i < lnth:
        yield pkkk[i]
        i +=1
        
act_l = []
prd_l = []

for i,j in enumerate(listgen(pkkk)):
    for k,l in zip(j[0][0], j[0][1]):
        act_l.append(k)
        prd_l.append(l)
diff_act_prd = [math.fabs(i-j) for i, j in zip(act_l, prd_l)]

# In[ ]:
def plot_hist(diff_act_prd):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(diff_act_prd, label='Absolute difference between True and Predicted MMSE Value')
    ax.set_title('Histogram absolute difference between True and Predicted MMSE Value')
    plt.ylabel('count')
    plt.xlabel('value')
    plt.show()
    fig.savefig('abs_pre_true.jpeg')

plot_hist(diff_act_prd)
print ('#######Finished#####')





