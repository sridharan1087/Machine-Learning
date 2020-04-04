import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers.tokenization_bert import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
import torch.optim as optim

class CustomSet(Dataset):
    def __init__(self,sentence,label,tokenizer,maxlen):
        self.sentence = sentence
        self.label = label
        self.tokenizer = tokenizer
        self.maxLen = maxlen
        
      
    def __len__(self):
        return len(self.sentence) 
    
    def __getitem__(self, index):
        t = self.tokenizer.encode_plus(self.sentence[index],
                                       max_length=self.maxLen,
                                       pad_to_max_length=True) 
        
        
        return {'input_ids':torch.tensor(t['input_ids']),
                'attn_mask':torch.tensor(t['attention_mask']),
                'label': torch.tensor(self.label[index])}
        
        
class sentimentAnalyzer(nn.Module):
    def __init__(self,freeze_bert=True):
        super(sentimentAnalyzer,self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        
        
        if freeze_bert == True:
            for p in self.model.parameters():
                p.requires_grad = False
                
                
        
        self.cls_layer = nn.Linear(1, 1)
        
        
    def forward(self,seq,attn_masks):
        cont_reps = self.model(seq, attention_mask = attn_masks)
        cls_rep = cont_reps[0][:, 0]
        logits = self.cls_layer(cls_rep)
        return logits
    
    
    
class fine_tune():
    def __init__(self):
        self.optimizer = optim.Adam(net.parameters(),lr=2e-05)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def training(self,n_epochs,j,net):
        for _ in range(n_epochs):
            for i in j:
                input_ids = i['input_ids']
                attn_mask = i['attn_mask']
                label = i['label']
                self.optimizer.zero_grad()
                tp = net(input_ids,attn_mask)
                print(tp)
                loss = self.loss_fn(tp.unsqueeze(1),label.float().unsqueeze(1))
                loss.backward()
                self.optimizer.step()
            
            print('epoch',_,loss)                 
            
        
    
    

        
        
        
if __name__=='__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = pd.read_csv(r'huggingFace\glue_data\SST-2\dev.tsv',
                     delimiter='\t')
    obj = CustomSet(df['sentence'],df['label'],tokenizer,30)
    j = DataLoader(obj,batch_size=1,num_workers=2)
    net = sentimentAnalyzer(freeze_bert=True)
    obj1 = fine_tune()
    obj1.training(5, j,net)
    
    
