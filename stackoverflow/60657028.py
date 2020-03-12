import pandas as pd

#Create a DataFrame
df = pd.DataFrame({'Invoice No':[123456,123457,123458,123459,123460,123461,123462,123464,123465],
                  'Date':['01-01-2020','02-01-2020','04-01-2020','06-01-2020','09-02-2020','09-02-2020','10-02-2020','11-02-2020','20-02-2020'],
                  'Description':['Rent Payment','Rent Payment','Rent Payment','Water Payment','Rent payment','Rep payment','Rep Payment','Rep Payment','Water Payment'],
                  'Vendor':['A']*9,
                  'Days':[0,1,2,2,34,0,1,2,11]})




description = None
Vendor = None
group = None
count = 0
a = []

#Loop over the dataframe 
for i, j in df.iterrows():
    if j['Description'].lower() == description and j['Vendor'].lower() == Vendor and j['Days']<=2:
        a.append(group)
    
    else:
        count+=1
        group = 'g'+str(count)
        print('')
        a.append(group)
    
    #Previous Description and Previous Vendors
    description = j['Description'].lower()
    Vendor = j['Vendor'].lower()    

#Columns group.  
df['group'] = a
print(df)

    
    
    