#How to use BertformaskedLM
from transformers import BertTokenizer, BertForMaskedLM
import torch

text = ' i have login [MASK], can you help me?'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
t = tokenizer.encode_plus(text)
print(t)

input_Sentence = torch.tensor(t['input_ids'])
segment_ids = torch.tensor(t['token_type_ids'])

model = BertForMaskedLM.from_pretrained('bert-base-uncased')

with torch.no_grad():
    output = model(input_Sentence.unsqueeze(0),token_type_ids = segment_ids.unsqueeze(0))
    prediction = output[0]


op = torch.argmax(prediction[0,5]).item()
print(tokenizer.decode(op))