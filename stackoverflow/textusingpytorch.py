import torch
import itertools

#Read the Input file
with open(r'data\1342-0.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

#split the words. op:[The,Project, of , Gutenberg,....]
data =[]
for line in lines:
    data.extend(line.split())
    

#Total size of vocabulary    
vocabSize = len(set(data))
print(f'VocabSize:{vocabSize}')


#Creating word to Index
word2Index = dict([(word.lower(),i) for i,word in enumerate(set(data))])
print(word2Index)

inputLine = """The Project Gutenberg EBook of Pride and Prejudice, by Jane Austen This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever.  You may copy it, give it away or
re-use it under the terms of the Project Gutenberg License included
with this eBook or online at www.gutenberg.org"""

d = []
for i in inputLine.split():
    print(i)
    print(word2Index[i.lower()])
    d.append(word2Index[i.lower()])
    
    
d = torch.tensor(d)
print(d)
t = torch.zeros(len(d),vocabSize)
print(t.shape)
Singleip = t.scatter_(1,d.unsqueeze(1),1)
print(Singleip.shape)