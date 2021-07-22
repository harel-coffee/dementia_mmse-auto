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

# In[1]:


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
from sklearn.model_selection import ParameterGrid
import pickle


# ## <a id='#00.2'>0.2. Define paths </a>

# In[2]:


data_path = '../../data/'


# # <a id='#01'>1. Data Cleaning and Preprocessing </a>

# In[3]:


patient_df = pd.read_csv(data_path+'patient_df_rnn.csv')

#print (patient_df.isna().sum()/patient_df.shape[0])
patient_df = patient_df.loc[:, patient_df.isna().sum()/patient_df.shape[0]< 0.4]

patient_df['patient_index'] = patient_df.index+1
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


# In[4]:


# pairs 
patient_indx_MMS = patient_df.groupby(['patient_id'])['patient_index', 'MINI_MENTAL_SCORE'].agg(lambda x : x.tolist())
patient_indx_MMS['count'] = patient_indx_MMS['patient_index'].apply(lambda x: len(x))


# # <a id='#02'>2. Deep Neural Network Model</a>

# In[95]:


MAX_LENGTH = 10
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print (device)
PAD_token = 6512


# In[96]:


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


def showPlot(points1, points2, file_name_suf):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=20)
    ax.yaxis.set_major_locator(loc)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(points1, label='train')
    plt.plot(points2, label='val')
    ax.legend()
    plt.savefig('./result/'+'train_val_batch'+file_name_suf+'.png')
    plt.close()
    

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def outputVar(indexes_batch):
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns padded input sequence tensor and lengths
def inputVar(indexes_batch):
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns all items for a given batch of pairs
def batch2TrainData(pair_batch):
    #print (pair_batch)
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch)
    output, mask, max_target_len = outputVar(output_batch)
    return inp, lengths, output, mask, max_target_len

def maskLoss(criterion,inp, target, mask):
    nTotal = mask.sum()
    #diff2 = (torch.flatten(inp) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
    #print (inp.shape, target.shape, mask.shape)
    #diff2 = torch.abs((torch.flatten(inp) - torch.flatten(target))) * torch.flatten(mask)
    #loss1 = torch.sum(diff2) / torch.sum(mask)
    loss  = criterion(inp, target.view(-1,1)).squeeze(1).masked_select(mask).mean()
    #print (loss1, loss)
    return loss, nTotal.item()

def readpicklefile(file):
    with open(file, 'rb') as f:
        file_obj = pickle.load(f)
    return file_obj

# In[97]:


class LSTMModel(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_weight, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        #self.dropout = dropout
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        #self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        #output = self.embedding_dropout(output)
        # print(output.dtype, hidden.dtype)
        # output = F.relu(output) # May be not converging due to this 
        # print(output.dtype, hidden.dtype)
        outputs, hidden = self.lstm(embedded, hidden)
        #print (outputs.shape)
        output = self.out(outputs[0])
        return output, hidden

    def initHidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device=device), torch.zeros(1, batch_size, self.hidden_size, device=device))

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
lstm_model_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

print(lstm_model_optimizer)'''
# In[98]:


def train(input_variable, lengths, target_variable, mask,  max_target_len, lstm_model,
                    lstm_model_optimizer, criterion, batch_size, clip, max_length=MAX_LENGTH):
    lstm_model_optimizer.zero_grad()
    
     # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    
    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0
    
    # hidden state
    lstm_model_hidden = lstm_model.initHidden(batch_size)

    for di in range(max_target_len):
        # print (input_tensor[di].dtype, lstm_model_hidden.dtype)
        lstm_model_output, lstm_model_hidden  = lstm_model(
            input_variable[di].view(1, -1), lstm_model_hidden)
        #print (lstm_model_output.dtype, target_tensor[di].dtype)
        #print (lstm_model_output, target_variable[di], mask[di])
        # print (lstm_model_output.shape, target_variable[di].shape, mask[di].shape)
        
        mask_loss, nTotal = maskLoss(criterion,lstm_model_output, target_variable[di], mask[di])
        #print (mask_loss, nTotal)
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal
        # loss += criterion(lstm_model_output[0], target_tensor[di], mask[di])
        #print (loss.requires_grad)
        #print ('train:',loss.dtype)
    
    loss.backward()
    
    _ = nn.utils.clip_grad_norm_(lstm_model.parameters(), clip)
    
    lstm_model_optimizer.step()

    return sum(print_losses)/ n_totals  # need to be changed

def evalu(input_variable, lengths, target_variable, mask,  max_target_len, lstm_model,
                criterion, batch_size=1, max_length=MAX_LENGTH):
    with torch.no_grad():
        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)
        
        # Initialize variables
        lstm_model_hidden = lstm_model.initHidden(batch_size)
        loss = 0
        print_losses = []
        n_totals = 0
        
        # Prediction
        output = []
        o_act = []
        o_pre = []
        
        for di in range(max_target_len):
            # print (input_tensor[di].dtype, lstm_model_hidden.dtype)
            lstm_model_output, lstm_model_hidden  = lstm_model(
                input_variable[di].view(1, -1), lstm_model_hidden)
            #print (lstm_model_output.dtype, target_tensor[di].dtype)
            #print (lstm_model_output, target_variable[di], mask[di])
            # print (lstm_model_output.shape, target_variable[di].shape, mask[di].shape)

            mask_loss, nTotal = maskLoss(criterion,lstm_model_output, target_variable[di], mask[di])
            #print (mask_loss, nTotal)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
            
            o_pre.append(target_variable[di].view(1, -1).flatten().cpu().numpy().tolist())
            o_act.append(lstm_model_output.flatten().cpu().numpy().tolist())
        o_act = np.array(o_act).flatten().tolist()
        o_pre = np.array(o_pre).flatten().tolist()
        output.append((o_act,o_pre))

    return (sum(print_losses)/ n_totals, output)


# In[99]:


def trainIters(train_pairs, val_pairs, lstm_model, n_iters, print_every, plot_every, batch_size, clip, learning_rate, optimser):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    plot_loss_val = []

    lstm_model_optimizer = optimser(lstm_model.parameters(), lr=learning_rate)
    
    
    # need to define loss function
    # criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss
    criterion = nn.L1Loss(reduce=False)
    

    for iter in range(1, n_iters+1):
        
        batches = batch2TrainData([random.choice(train_pairs) for _ in range(batch_size)])
        input_variable, lengths, target_variable, mask, max_target_len = batches
    
        '''print (input_variable)
        print (lengths)
        print (mask)
        print (max_target_len)
        #print (target_variable)'''
        
        loss = train(input_variable, lengths, target_variable, mask,  max_target_len, lstm_model,
                    lstm_model_optimizer, criterion, batch_size, clip)
        
        #print ('trainIters:',loss.dtype)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            total_val_loss = 0
            lstm_model.eval()
            output_total = []
            val_size = len(val_pairs)
            
            for i in range(0, val_size):
                batches_val = batch2TrainData([val_pairs[i]])
                input_variable_val, lengths_val, target_variable_val, mask_val, max_target_len_val = batches_val
                loss1, output = evalu(input_variable_val, lengths_val, target_variable_val, mask_val,  max_target_len_val, lstm_model,
                criterion)
                total_val_loss+=loss1
                output_total.append(output)
            
            print_val_loss_avg = total_val_loss / val_size
            
            print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg, print_val_loss_avg))
            
            plot_loss_val.append(print_val_loss_avg)
            
            lstm_model.train()      

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
            
    return (plot_losses, plot_loss_val, output_total)


# In[101]:


def listgen(pkkk):
    lnth = len(pkkk)
    i = 0
    while i < lnth:
        yield pkkk[i]
        i +=1

def act_prd_fun(output_total, file_name_suf):
    act_prd_l=[]
    for i,j in enumerate(listgen(file_o)):
        if i ==5:
            break
    act_prd_l.append(list(j[0]))
    return act_prd_l


def diff_act_prd_func(output_total, file_name_suf):
    with open('./result/'+'output_total_batch'+file_name_suf+'.pkl', 'wb') as f:
        pickle.dump(output_total, f)
        
    act_l = []
    prd_l = []
    for i,j in enumerate(listgen(output_total)):
        for k,l in zip(j[0][0], j[0][1]):
            act_l.append(k)
            prd_l.append(l)
    return [math.fabs(i-j) for i, j in zip(act_l, prd_l)]

def plot_hist(diff_act_prd, file_name_suf):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(diff_act_prd, label='Absolute difference between True and Predicted MMSE Value')
    ax.set_title('Histogram absolute difference between True and Predicted MMSE Value')
    plt.ylabel('count')
    plt.xlabel('value')
    fig.savefig('./result/'+'abs_pre_true_batch'+file_name_suf+'.jpeg')
    
    

hidden_size = 171
output_size = 1
token_size = (1,171)
epoch = 40

#token_emb_init, clip, batch_size, learning rate,
param_grid = {'batch_size':[16], 'clip':[10], 'learning_rate':[0.001], #clip can be [5,10]
              'optimser':[optim.Adam], 'pad_token_init':[torch.zeros]}



for params in list(ParameterGrid(param_grid)):
    
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    
    print ('Dataset Size:', len(train_pairs), len(val_pairs))
    
    with open('./result/'+'train'+file_name_suf+'.pkl', 'wb') as f:
        pickle.dump(train_pairs, f)
        
    with open('./result/'+'test'+file_name_suf+'.pkl', 'wb') as f:
        pickle.dump(val_pairs, f)
    
    print (params)
    pad_token_init = params['pad_token_init'](token_size, )
    batch_size = params['batch_size']
    clip = params['clip']
    learning_rate = params['learning_rate']
    optimser = params['optimser']
    
    # embedding Matrix
    embedding_weight = torch.cat((torch.from_numpy(patient_df.iloc[:, 4:].values).float(), pad_token_init), dim=0)
    # embedding_weight = F.normalize(torch.from_numpy(patient_df.iloc[:, 4:].values).float(), p=2, dim=1, eps=1e-12, out=None)
    print(embedding_weight.shape)

    lstm_model = LSTMModel(hidden_size, output_size, embedding_weight, dropout=0.0).to(device)
    #trainIters(lstm_model, 3, print_every=1)
    plot_losses, plot_loss_val, output_total = trainIters(train_pairs, val_pairs, lstm_model, int(len(train_pairs)/batch_size)*epoch, 
                                                          int(len(train_pairs)/batch_size), int(len(train_pairs)/batch_size), 
                                                          batch_size, clip, learning_rate, optimser)
    
    #plot_losses, plot_loss_val, output_total = trainIters(train_pairs, val_pairs, lstm_model, 1, 
    #                                                      1, 1, 
    #                                                      batch_size, clip, learning_rate, optimser)
    
    file_name_suf = '_{}_{}_{}_{}_{}'.format(params['pad_token_init'].__name__, batch_size, clip, learning_rate, optimser)
    diff_act_prd = diff_act_prd_func(output_total, file_name_suf)
    plot_hist(diff_act_prd, file_name_suf)
    showPlot(plot_losses, plot_loss_val, file_name_suf)
    print ('diff_act_prd', len(diff_act_prd))
    


# # <a id='#03'>3. Result Analysis</a>

# In[ ]:





# In[ ]:




