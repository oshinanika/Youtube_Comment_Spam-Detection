import pandas as pd
import numpy as np


dataset=pd.read_csv("..\..\Datasets\YouTube-Spam-Collection-v1\KatyPerry.csv")

#csv, in columns
label=dataset['CLASS']
#print(label)
text=dataset['CONTENT']
#print(text)



#whole document is converted into a array
label=np.array(label)
text=np.array(text)