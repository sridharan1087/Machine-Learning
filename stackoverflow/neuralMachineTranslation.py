#coding=latin-1 
import cProfile
import numpy as np
import json 
import pickle
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import LSTM

class readingAndIndexingCharacters(object):
    
    def __init__(self):
        self.inputs = []
        self.targets = []
    
    def indexingCharacters(self):
        
        #Open and read lines. Split it by tab. 
        with open(r'data/spa-eng/spa_test.txt','r',encoding='utf-8') as f:
            for lines in f.readlines():
                try:
                    sentence = lines.split('\t')
                    inpt,target = sentence[0],sentence[1]
                    self.inputs.append(inpt)
                    self.targets.append(target)
                except Exception as e:
                    print(f'Exception while reading lines:{e}')
             
        print(f'input: {self.inputs[0:10]}') 
        print(f'targe: {self.targets[0:10]}')  
        
        
        #Reading unique characters from the inputs and target.
        inputCharacterIndex = set()
        targetCharacterIndex = set()
        
        for i,j in zip(self.inputs,self.targets):
            for char in i:
                try:
                    inputCharacterIndex.add(char)
                except Exception as e:
                    print(f'Exception in reading characters from input: {e}')
                    
            for char in j:
                try:
                    targetCharacterIndex.add(char)
                except Exception as e:
                    print(f'Exception in reading characters from target: {e}')
                    
        #Indexing the characters.
        inputTokenIndex = {}
        targetTokenIndex = {}
        
        #Creating Index of Input and Target Data.           
        for k,(i,j) in enumerate(zip(inputCharacterIndex,targetCharacterIndex)):
            inputTokenIndex[i] = k
            targetTokenIndex[j] = k
            
        #Write to a json file
        with open(r'data/inputToken.json','w') as inputJson:
            json.dump(inputTokenIndex,inputJson)
            
        with open(r'data/targetToken.json','w') as targetJson:
            json.dump(targetTokenIndex,targetJson)
            
        nmtDetails = {'noOfInputs' : len(self.inputs),
                   'maxLenInputs' : max([len(txt) for txt in self.inputs]),
                   'maxLenTargets' : max([len(txt) for txt in self.targets]),
                   'maxCharInputLen' : len(inputCharacterIndex),
                   'maxCharTargetLen' : len(targetCharacterIndex)}
        
        details = open(r'data/nmtDetails.pckl','ab')
        pickle.dump(nmtDetails,details)
        
       
    def trainingData(self):
        #Load input and target json
        with open(r'data/targetToken.json','r') as f:
            TargetjsonData = json.load(f)
            
        with open(r'data/inputToken.json','r') as f:
            inputJsonData = json.load(f)
            
        nmtDetails = open(r'data/nmtDetails.pckl','rb')
        nmtDetails = pickle.load(nmtDetails)
        print(json.dumps(nmtDetails,indent=4,sort_keys=True))

        inputs = np.zeros((nmtDetails['noOfInputs'],
                           nmtDetails['maxLenInputs'],
                           nmtDetails['maxCharInputLen']
                           ))
         
         
        targets = np.zeros((nmtDetails['noOfInputs'],
                           nmtDetails['maxLenTargets'],
                           nmtDetails['maxCharTargetLen']
                           ))
        
        
        for i,(inpt,target) in enumerate(zip(self.inputs,self.targets)):
            for t,char in enumerate(inpt):
                inputs[i,t,inputJsonData[char]]=1
                
            for t,char in enumerate(target):
                targets[i,t,TargetjsonData[char]]=1
                
        print(f'Input-shape: {inputs.shape}')    
        print(f'Target-shape: {targets.shape}')
        return({'input':inputs,'target':targets}) 
       
    def modelArch(self,x,y):
        try:
            model =Sequential()
            model.add(LSTM(30,input_shape=(5,22)))
            model.add(Dense(35,activation="softmax"))
            
            model.compile('Adam',loss="categorical_crossentropy",metrics=['accuracy'])
            model.fit(x,y)
            
            
        except Exception as e:
            print(f'Exception in model Architecture: {e}')
                
    
if __name__=='__main__':
        obj = readingAndIndexingCharacters()
        #obj.indexingCharacters()
        trainTestData = obj.trainingData()
        obj.modelArch(trainTestData['input'], trainTestData['target'])
        
        
        