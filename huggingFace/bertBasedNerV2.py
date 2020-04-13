!pip install -q keras
!pip install -q transformers
!pip install -q seqeval
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification
import torch.nn as nn
import torch


data = pd.read_csv(io.BytesIO(uploaded['small_ner_dataset.csv']),encoding='latin1').fillna(method="ffill")
data.tail(10)
print(data.columns[0])

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby(self.data.columns[0]).apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
        
        
getter = SentenceGetter(data)
sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
sentences[0]

labels = [[s[2] for s in sent] for sent in getter.sentences]
print(labels[0])


tags_vals = list(set(data["Tag"].values))
tag2idx = {t: i for i, t in enumerate(tags_vals)}
print('**************',tag2idx)

MAX_LEN = 75
bs = 32

from seqeval.metrics import f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print(tokenized_texts[0])

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")



tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")


attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

print(tr_inputs.shape)
print(tr_tags.shape)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)


model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))
model.to('cuda')
#param_optimizer = list(model.classifier.named_parameters()) 
#optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay_rate': 0.01},{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay_rate': 0.0}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

epochs = 30
max_grad_norm = 1.0
# loss = nn.mse()

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to('cuda') for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        #print('#####################',b_input_mask.shape)
        #print('*********************',b_labels.shape)
        
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss[0].backward()
        # track train loss
        tr_loss += loss[0].item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to('cuda') for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
          tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
          logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
          
        logits = logits[0].detach().cpu().numpy()
        print(logits[0].shape)
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        print('prediction',len(predictions))
        true_labels.append(label_ids)
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_loss += tmp_eval_loss[0].mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    print('pred_tags',pred_tags)
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print('valid tags',valid_tags)
    #print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


model.eval()
text = "london and iraq meet yesterday"

t1 = tokenizer.encode(text)
print(tokenizer.decode(t1))
print(t1)
t = torch.tensor(t1).to('cuda')

with torch.no_grad():
  output = model(t.unsqueeze(1))
  prediction = output[0]

d = [i for i in torch.argmax(prediction,axis=2)]

def get_key(val): 
    for key, value in tag2idx.items(): 
         if val == value: 
             return key 
for i in range(len(d)):
  print(tokenizer.decode(t1[i]),'--->',get_key(d[i].item()))