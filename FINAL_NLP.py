import sys
sys.path.append("D:/python/Lib/site-packages")
import time

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd,  string
import numpy as np
import csv

file = open("C://Users//pc//Desktop//training-v1//training-v1//offenseval-training-v1.tsv",encoding="utf8")
data=file.read()
labels, texts = [], []

for line in data.split("\n"):
    if line=="":
      break
    content = line.split("\t")
    labels.append(content[2])
    temp=content[1].replace("@USER","")
    texts.append(temp)
labels=labels[1:]
texts=texts[1:]
i=0
for s in texts :
    s=s.lower()
    s = ''.join([i for i in s if not i.isdigit()])
    texts[i]=s
    i+=1
    
# create a dataframe using texts and lables
trainDF = pd.DataFrame()
trainDF['label'] = labels
trainDF['text'] = texts

trainDF.loc[trainDF['label']=='OFF','label']=1
trainDF.loc[trainDF['label']=='NOT','label']=0
df_x=trainDF['text']
df_y=trainDF['label']
cv = TfidfVectorizer(min_df=1,stop_words='english')

# split the dataset into training and validation datasets 
x_test = open("C://Users//pc//Desktop//testset-taska.tsv",encoding="utf8")
#for evaluation purpose
xx_train, xx_test, yy_train, yy_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

x_traincv=cv.fit_transform(df_x)
xx_traincv=cv.fit_transform(xx_train)
xx_testcv=cv.transform(xx_test)
xx_testcv=xx_testcv.toarray()
xx_testcv=xx_testcv[0:]


test_data=x_test.read()
ids,test_label, test_texts ,testprint = [], [],[],[]

for line in test_data.split("\n"):
    if line=="":
      break
    content = line.split("\t")
    ids.append(content[0])
    testprint.append(content[1])
    temp=content[1].replace("@USER","")
    test_texts.append(temp)
ids=ids[1:]
test_texts=test_texts[1:]
i=0
for s in test_texts :
    s=s.lower()
    s = ''.join([i for i in s if not i.isdigit()])
    test_texts[i]=s
    i+=1

# create a dataframe using texts and lables
testDF = pd.DataFrame()
testDF['ids'] = ids
testDF['test_texts'] = test_texts
df_x=testDF['test_texts']
df_y=trainDF['label']
x_testcv=cv.transform(test_texts)
x_testcv=x_testcv.toarray()

y_train=df_y.astype('int')
yy_train=yy_train.astype('int')
a=np.array(yy_test)
a= a.astype('int')
b=np.array(xx_testcv)
b= b.astype('int')

#Naive Bayes
mnb = MultinomialNB(alpha = 0.1, fit_prior = True)
y_train=df_y.astype('int')
yy_train=yy_train.astype('int')
#mnb.fit(x_traincv,y_train)
mnb.fit(xx_traincv,yy_train)
predictions=mnb.predict(x_testcv)
predictions2=mnb.predict(xx_testcv)
count=0
for i in range (len(predictions2)):
    if predictions2[i]==a[i]:
        count=count+1
print('Naive bayes')
print(predictions)
print('Accuracy: ')
print(count/len(predictions2))
#accuracy = cross_val_score(mnb, b, a, cv = 10, scoring='accuracy').mean()
#print('Overall for K-fold accuracy: {} %'.format(accuracy*100))
conf_mat = confusion_matrix(predictions2, a)
print('Confusion Matrix: ')
print(conf_mat)
testvv = pd.DataFrame()
testvv['label'] = predictions
testvv['text'] = testprint[1:]

testvv.loc[testvv['label']==1,'label']='OFF'
testvv.loc[testvv['label']==0,'label']='NOT'
print(testvv)


#logistic Regression
lr = LogisticRegression(penalty='l2' ,C=10 ,solver = 'lbfgs', max_iter=1000)
#lr.fit(x_traincv,y_train)
lr.fit(xx_traincv,yy_train)
predictions=lr.predict(x_testcv)
predictions2=lr.predict(xx_testcv)
count=0
for i in range (len(predictions2)):
    if predictions2[i]==a[i]:
        count=count+1
print('Logistic Reggresion')
print(predictions)
print('Accuracy:')
print(count/len(predictions2))
conf_mat = confusion_matrix(predictions2, a)
print('Confusion Matrix: ')
print(conf_mat)
#accuracy = cross_val_score(lr, b, a, cv =10, scoring='accuracy').mean()
#print('Overall for K-fold accuracy: {} %'.format(accuracy*100))

testvv = pd.DataFrame()
testvv['label'] = predictions
testvv['text'] = testprint[1:]

testvv.loc[testvv['label']==1,'label']='OFF'
testvv.loc[testvv['label']==0,'label']='NOT'
print(testvv)



#KNN
knn=KNeighborsClassifier(n_neighbors=1)
#knn.fit(x_traincv,y_train)
knn.fit(xx_traincv,yy_train)
predictions= knn.predict(x_testcv) 
predictions2= knn.predict(xx_testcv)
count=0
for i in range (len(predictions2)):
    if predictions2[i]==a[i]:
        count=count+1
print('first KNN')
print(predictions)
print('Accuracy:')
print(count/len(predictions2))
conf_mat = confusion_matrix(predictions2, a)
print('Confusion Matrix: ')
print(conf_mat)
#accuracy = cross_val_score(knn, b, a, cv = 10, scoring='accuracy').mean()
#print('Overall for K-fold accuracy:{} %'.format(accuracy*100))


testvv = pd.DataFrame()
testvv['label'] = predictions
testvv['text'] = testprint[1:]
testvv.loc[testvv['label']==1,'label']='OFF'
testvv.loc[testvv['label']==0,'label']='NOT'
print(testvv)

#KNN2
#myList = list(range(1,10))
### subsetting just the odd ones
#neighbors = list(filter(lambda x: x % 2 != 0, myList))
#print(neighbors)
### empty list that will hold cv scores
#cv_scores = []

### perform 10-fold cross validation
#for k in neighbors:
    #knn = KNeighborsClassifier(n_neighbors=k)
    #scores = cross_val_score(knn, x_traincv, y_train, cv=10, scoring='accuracy')
    #cv_scores.append(scores.mean())
#MSE = [1 - x for x in cv_scores]   
#optimal_k = neighbors[MSE.index(min(MSE))]
#print(optimal_k)
#knn=KNeighborsClassifier(n_neighbors=optimal_k)
#knn.fit(x_traincv,y_train)
#knn.fit(xx_traincv,yy_train)
#predictions= knn.predict(x_testcv) 
#predictions2= knn.predict(xx_testcv)
#print('second knn')
#print(predictions)
#count=0
#for i in range (len(predictions2)):
    #if predictions2[i]==a[i]:
        #count=count+1
#print('Accuracy:')
#print(count/len(predictions2))
#accuracy = cross_val_score(knn, b, a, cv = 10, scoring='accuracy').mean()
#print('Overall accuracy: {} %'.format(accuracy*100))
#conf_mat = confusion_matrix(predictions2, a)
#print('Confusion Matrix: ')
#print(conf_mat)
#testvv = pd.DataFrame()
#testvv['label'] = predictions
###testvv['text'] = csv.reader(x_test,delimiter="\n")
#testvv['text'] = testprint[1:]

#testvv.loc[testvv['label']==1,'label']='OFF'
#testvv.loc[testvv['label']==0,'label']='NOT'
#print(testvv)


#SVM 
model = svm.SVC(C=1, kernel='linear', gamma=1) 
#model.fit(x_traincv,y_train)
model.fit(xx_traincv,yy_train)
#Predict Output
predictions= model.predict(x_testcv)
predictions2= model.predict(xx_testcv)
count=0
for i in range (len(predictions2)):
    if predictions2[i]==a[i]:
        count=count+1
print('SVM')
print(predictions)
print('Accuracy:')
print(count/len(predictions2))
#accuracy = cross_val_score(model, b, a, cv = 10, scoring='accuracy').mean()
#print('Overall for K-fold accuracy:: {} %'.format(accuracy*100))
conf_mat = confusion_matrix(predictions2, a)
print('Confusion Matrix: ')
print(conf_mat)
testvv = pd.DataFrame()
testvv['label'] = predictions
testvv['text'] = testprint[1:]
testvv.loc[testvv['label']==1,'label']='OFF'
testvv.loc[testvv['label']==0,'label']='NOT'
print(testvv)


## Decision Tree
model = tree.DecisionTreeClassifier(criterion='gini') 
# Train the model using the training sets and check score
#model.fit(x_traincv,y_train)
model.fit(xx_traincv,yy_train)
#Predict Output
predictions= model.predict(x_testcv)
predictions2= model.predict(xx_testcv)
count=0
for i in range (len(predictions2)):
    if predictions2[i]==a[i]:
        count=count+1
print('Decision Tree')
print(predictions)
print('Accuracy:')
print(count/len(predictions2))
#accuracy = cross_val_score(model, b, a, cv = 10, scoring='accuracy').mean()
#print('Overall for K-fold accuracy: {} %'.format(accuracy*100))
conf_mat = confusion_matrix(predictions2, a)
print('Confusion Matrix: ')
print(conf_mat)
testvv = pd.DataFrame()
testvv['label'] = predictions
testvv['text'] = testprint[1:]

testvv.loc[testvv['label']==1,'label']='OFF'
testvv.loc[testvv['label']==0,'label']='NOT'
print(testvv)




# Create Random Forest object
model= RandomForestClassifier(n_estimators=1000)
#Train the model using the training sets and check score
model.fit(x_traincv,y_train)
model.fit(xx_traincv,yy_train)
#Predict Output
predictions= model.predict(x_testcv)
predictions2= model.predict(xx_testcv)
count=0
for i in range (len(predictions2)):
   if predictions2[i]==a[i]:
       count=count+1
print('Random Forest')
print(predictions)
print('Accuracy:')
print(count/len(predictions2))
conf_mat = confusion_matrix(predictions2, a)
print('Confusion Matrix: ')
print(conf_mat)