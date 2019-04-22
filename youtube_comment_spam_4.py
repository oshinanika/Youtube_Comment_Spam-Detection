from youtube_comment_spam_3 import *
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

clf=XGBClassifier(n_estimators=500)

clf.fit(x_train_trans, y_train)
y_pred=clf.predict(x_test_trans)

print(y_pred)
print(y_test)

print("ACCURACY: ")

print(accuracy_score(y_test,y_pred))