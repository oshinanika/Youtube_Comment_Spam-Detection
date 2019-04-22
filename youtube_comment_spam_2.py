from youtube_comment_spam_1 import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


x_train=[]
x_test=[]
y_train=[]
y_test=[]

label, text = shuffle(label,text)

x_train,x_test,y_train,y_test=train_test_split(text,label,train_size=0.9)

x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)


#print(y_test)