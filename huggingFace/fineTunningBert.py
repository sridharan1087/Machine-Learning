import torch
import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer
from transformers.modeling_bert import BertModel


class SstDataSet(object):
    def __init__(self,fileName,maxLen):
        self.df = pd.read_csv(fileName,delimiter='\t')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.maxLen = maxLen
        
        
    def __getitem__(self,index):
        try:
            sentence = self.df.loc[index,'sentence']
            label = self.df.loc[index,'label']
            
            tokenizedText = self.tokenizer.tokenize(sentence)  # For Indexing of the words
            print(f'Tokenized Text',tokenizedText)
            
            tokenizedText = ['[cls]']+tokenizedText+['[sep]']
            
            if len(tokenizedText) < self.maxLen:
                tokenizedText += ['[PAD]' for _ in range(self.maxLen-len(tokenizedText))]
                
            else:
                tokenizedText = tokenizedText[:self.maxLen]+['[SEP]']
                
            tokens = self.tokenizer.convert_tokens_to_ids(tokenizedText)
            tokenIds = torch.tensor(tokens)
            
            attn_mask = (tokenIds != 0).long()
            
            return tokenIds,attn_mask, label           
            
        except Exception as e:
            print('Exception in __Getitem__',e)
            
            


from torch.utils.data import DataLoader
val_set = SstDataSet(r'glue_data\SST-2\dev.tsv',maxLen=30)
train_set = SstDataSet(r'glue_data\SST-2\train.tsv',maxLen=30)


train_loader = DataLoader(train_set, batch_size = 64, num_workers = 5)
val_loader = DataLoader(val_set, batch_size = 64, num_workers = 5)
 

import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self,freeze_bert=True):
        super(SentimentClassifier,self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad=False
                
        
        self.cls_layer = nn.Linear(768,1)

    def forward(self,seq, attn_masks):
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)
        cls_rep = cont_reps[:, 0]
        logits = self.cls_layer(cls_rep)
        
        return logits
    
    


net = SentimentClassifier(freeze_bert=True)
import torch.optim as optim


loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(),lr = 1e-04)

def training(epochs,loss_fn,optimizer,train_loader,val_loader):
    for i in range(epochs):
        for it, (seq,attn_mask,label) in enumerate(train_loader):
            optimizer.zero_grad()
            model = net(seq,attn_mask)
            loss = loss_fn(model.squeeze(-1),label)
            loss.backward()
            optimizer.step()
            print('Epoch1')
            
            
            
training(1, loss_fn, optimizer, train_loader, val_loader)
        
        