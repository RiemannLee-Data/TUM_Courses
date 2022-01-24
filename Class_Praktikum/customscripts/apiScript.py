import requests 
import pandas as pd
import csv
import time
import glob
import os
import pandas as sd
path = r'C:\Users\Sreejit\segdata'

cList= ["842", "276", "392", "344", "826" , "381", "528", "699" , "484", "124"]
j=0
nm=0
while j < len(cList):
    time.sleep(1)
    url="http://comtrade.un.org/api/get?r="+ cList[j]+"&px=HS&ps=2019&p=ALL&cc=TOTAL&type=C&freq=M&fmt=csv&rg=1,2"
    df = pd.read_csv(url) 
    df.to_csv(os.path.join(path,r'outcom'+str(nm)+'.csv'))
    nm=nm+1
    j=j+1
    
path =  r'C:\Users\Sreejit\segdata'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = sd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = sd.concat(li, axis=0, ignore_index=True)
frame.to_csv(os.path.join(path,r'final2.csv'))



path =  r'C:\Users\Sreejit\segdata'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = sd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = sd.concat(li, axis=0, ignore_index=True)
frame.to_csv(os.path.join(path,r'final2.csv'))