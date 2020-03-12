import pandas as pd

df = pd.DataFrame({'a':[1,2,3,4],'b':[5,6,7,8],'c':[9,0,11,12]},index=[0,3,5,6])
df1 = pd.DataFrame({'a':[13,14,15,16],'b':[17,18,19,20],'c':[21,22,23,24]},index=[1,2,4,7])

df2 = df.append(df1)
print(df2.sort_index())
