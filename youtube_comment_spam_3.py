from youtube_comment_spam_2 import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

count_vect= CountVectorizer(decode_error='ignore') #if there is a characted outside of unicoding.. the decode error will be ignored

#train(fit) and transform the training set
#count the terms
x_train_count=count_vect.fit_transform(x_train)

#frequency of each term in each document
tfidf_trans=TfidfTransformer()
x_train_trans=tfidf_trans.fit_transform(x_train_count)

#learns from the training set to be applied to the test set.. so no need for fit
#count_vect is trained now so use it on test set
x_test_count=count_vect.transform(x_test)

#tfidf_trans is trained now
x_test_trans=tfidf_trans.transform(x_test_count)
 #print(x_test_trans)