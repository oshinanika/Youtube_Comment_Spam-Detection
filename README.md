# Youtube_Comment_Spam_Detection

This is a Spam Detection Project that works on a collection of youtube comments from various youtube videos.

## Getting Started

In this project we will be using

* Python
* pandas, sklearn, numpy
* Dataset from a .csv file
* Boosting Algorithm

### Prerequisites

The collection of youtube comments we'll be using is from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection) which is in a .csv format.[Details on using .csv format](https://www.shanelynn.ie/python-pandas-read_csv-load-data-from-csv-files/)

### Load dataset and convert into array

We use `pandas` library to load the .csv file and then separate the labels and messages.

```
dataset=pd.read_csv("..\YouTube-Spam-Collection-v1\KatyPerry.csv")

label=dataset['CLASS']
text=dataset['CONTENT']

```
Then we need to convert these columns to arrays for easier calculation using `numpy` library.

### Split the dataset into Training and Testing set

We will be connecting python scripts using ...
```
from script_name import *

```
Now, the dataset we have, needs to be split into **Training** and **Testing** sets using the `sklearn` library and we will use 90% of the dataset for training our model .But before that, we can shuffle the `label` and `text` arrays to mix up the data.

```
label, text = shuffle(label,text)

x_train,x_test,y_train,y_test=train_test_split(text,label,train_size=0.9)
```
### Preparing the Text Data for modeling

To properly know the techniques to prepare any text before applying predictive models on them ...
* [MonkeyLearn](https://monkeylearn.com/blog/word-embeddings-transform-text-numbers/)
* [MachineLearningMastery](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/)

We are using `Countvectorizer` class API to tokenize the important texts ...
```
count_vect= CountVectorizer(decode_error='ignore')
x_train_count=count_vect.fit_transform(x_train)

```
and `TfidfTransformer` to calculate the weight of these words in the **Training** set.
```
tfidf_trans=TfidfTransformer()
x_train_trans=tfidf_trans.fit_transform(x_train_count)

```
Now we will apply this learned model onto the **Test** set to find the important words to identify spam comments.
```
x_test_count=count_vect.transform(x_test)
x_test_trans=tfidf_trans.transform(x_test_count)
```
### Boosting Our Algorithm for better performance

* [Boosting Algorithms](https://hackernoon.com/boosting-algorithms-adaboost-gradient-boosting-and-xgboost-f74991cad38c)
* [Installation](https://xgboost.readthedocs.io/en/latest/)
* [Tuning](https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/)

We apply the boosting algorithm on the transformed **Test** set after fitting the transformed **Training** set to this algorithm
```
clf.fit(x_train_trans, y_train)
y_pred=clf.predict(x_test_trans)

```
Finally we compare the predicted labels(spam(1) or no spam(0)) of the boosting algorithm to the original labels from the **Test** set to get the accuracy score.
```
accuracy_score(y_test,y_pred)
```
At **n_estimator=200** we get almost 97% accuracy through this model.

## Built With

* Sublime Text 3
* Python 3.3

## References
* https://hackernoon.com/a-simple-spam-classifier-193a23666570


