from transformers import BertTokenizer,BertModel
import torch

text = "i have login issue"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizedText = tokenizer.tokenize(text)
print(f'Tokenized Text',tokenizedText)

tokenizedText[-1] = 'MASK'

inputIds = tokenizer.convert_tokens_to_ids(tokenizedText)
print(f'Input Ids', inputIds)

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

tokenTensor = torch.tensor(inputIds)
print(f'TokenTensor',tokenTensor)

output = model(tokenTensor.unsqueeze(0))
print(output[0])
prediction = torch.argmax(output[0])
print(prediction)
print(tokenizer.convert_ids_to_tokens([prediction]))
