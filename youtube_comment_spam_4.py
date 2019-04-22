from youtube_comment_spam_3 import *
#Gradient Boosting algorithm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#n_estimators are level of decision tree
clf=XGBClassifier(n_estimators=500)

#apply the labels based on the tfidftransformer on testing set
clf.fit(x_train_trans, y_train)
y_pred=clf.predict(x_test_trans)

print(y_pred)
print(y_test)

print("ACCURACY: ")

print(accuracy_score(y_test,y_pred))