from youtube_comment_spam_2 import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

count_vect= CountVectorizer(decode_error='ignore')
x_train_count=count_vect.fit_transform(x_train)

tfidf_trans=TfidfTransformer()
x_train_trans=tfidf_trans.fit_transform(x_train_count)

x_test_count=count_vect.transform(x_test)
x_test_trans=tfidf_trans.transform(x_test_count)
print(x_test_trans)