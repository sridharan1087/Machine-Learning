import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers.tokenization_bert import BertTokenizer

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
        
        
        return torch.tensor(t['input_ids'])
    
    
    
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = pd.read_csv(r'C:\Users\ajay\Documents\GitHub\Machine-Learning\huggingFace\glue_data\SST-2\dev.tsv',
                     delimiter='\t')
    obj = CustomSet(df['sentence'],df['label'],tokenizer,30)
    j = DataLoader(obj,batch_size=5,num_workers=2)
    for i in j:
        print(i)
