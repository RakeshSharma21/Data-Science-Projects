#!/usr/bin/python


### importing the libraries needed for project implementation
import sys
import pickle
from feature_format import featureFormat, targetFeatureSplit
import pprint as pp
from tester import dump_classifier_and_data
from tester import test_classifier
from sklearn.pipeline import  Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot
from operator import itemgetter


### set the path where python will look for files while reading or writing.

sys.path.append("../tools/")


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary'] # You will need to use more features

pp.pprint ("-------------------------------Task 1:---------------------------------------")
pp.pprint ("--------------------------Feature Selection----------------------------------")


### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    
### Lets select all the feature for one of the point of interest person

for key, values in data_dict.items():
    if key == 'LAY KENNETH L':
        for k,v in values.items():
            if k not in ('poi','salary'):
                features_list.append(k)


### declaration for poi and non poi list

poi_list=[]
non_poi_list=[]

### identifying poi and non poi members from the dataset.

for key, values in data_dict.items():
    for k,v in values.items():
        if k =='poi':
            if v:
                poi_list.append(k)
            else:
                non_poi_list.append(k)
                

### Dictionary declaration to store total number of NaN per feature

feature_with_missing_val_cnt={}


#### Identifying total number of NaN per feature.

for key, values in data_dict.items():
    for k,v in values.items():
        if v =='NaN':
            if k in feature_with_missing_val_cnt:
                feature_with_missing_val_cnt[k]+=1
            else:
                feature_with_missing_val_cnt[k]=1
                

### Total number of NaN per features

pp.pprint("The dictionary below shows the number of NaN values per feature:" )
pp.pprint (feature_with_missing_val_cnt)


### Count the total number for features select for one the the poi.         

print("Total number of features: ",len(features_list))

### Total number of data datapoints in given dataset.

print("Total number of data points: ",len(data_dict))

###Number of employees which falls in the list of poi

print("Total number of POIs in the dataset: ",len(poi_list))

### Number of employees which falls in the list of non poi

print("Total number of NonPOIs in the dataset: ",len(non_poi_list))


### Task 2: Remove outliers
pp.pprint ("-------------------------------Task 2:---------------------------------------")
pp.pprint ("--------------------------Remove outliers------------------------------------")




#### By going through the dataset and intial analysis it seems pretty clear that 'TOTAL' and'
## 'THE TRAVEL AGENCY IN THE PARK' are not any employee

### Total outlier can also be observed by ploting salaries and bonus of the employees on graph
test_features = ["salary", "bonus"]
data = featureFormat(data_dict, test_features)

### visualizing the salary vs bonus plot.

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### Total number of data points before removing outlier

print ("Total number of data points in data dictionary before outlier removal: ",len(data_dict))


### Removing outliers from the data dictionary itself

outlier=['TOTAL','THE TRAVEL AGENCY IN THE PARK']

### person set to collect all the employees in the dataset

person =set()

### collecting all the employees in the person dataset.

for key, values in data_dict.items():
    person.add(key)

### removing the outlier from the the dataset.
    
for key in person:
    if key in outlier:
        data_dict.pop(key,0)

### remove the employee who is having the the values as NaN

data_dict.pop('LOCKHART EUGENE E',0)

### Number of data point after removal of outliers.

print ("Total number of data points in data dictionary after outlier removal: ",len(data_dict))


### Task 3: Create new feature(s)
print ("-------------------------------Task 3:---------------------------------------")
print ("--------------------------Create new features------------------------------------")

### dumping data_dict in to my_dataset.

my_dataset = data_dict

###I had an intuition that bonus/total_payment could be a useful feature as bonus/total_payments should be relatively higher for 
###a poi than a normal employee. and instead of using bonus and total_payment as different feature it is always a good idea to 
### use the single feature that can tell the story 

###Another good feature can be to identifying poi to and from mail to the total number of email send or receive by an employee
### my intuition is a poi would send or receive most of the email from poi and to poi than to the email from `non pois

### based on the intuition idenfying all the concerned features.

concernedFeat=['total_payments', 'bonus','from_this_person_to_poi', 'from_poi_to_this_person', 'to_messages', 'from_messages']

### Formating data for concerned features.
dict_data_1 = featureFormat(my_dataset, concernedFeat, sort_keys = True)

### Adding a new feature bonus/total_payment

### Handling the float nan values.

import math
def check_Nan(val):
    if math.isnan(val):
        return 0
    else:
        return val

### Adding two new feature bonus/total_payment and poi to and from email to total of to and from email.

for key,values in data_dict.items():
    if data_dict[key]['total_payments']!=0:
        data_dict[key]['bonus_to_tot_pay_ratio']=check_Nan(float(data_dict[key]['bonus'])/float(data_dict[key]['total_payments']))
    if data_dict[key]['to_messages']!=0 and data_dict[key]['from_messages']!=0:
        data_dict[key]['poi_to_total_email_ratio']=check_Nan(float(data_dict[key]['from_this_person_to_poi'])+float(data_dict[key]['from_poi_to_this_person'])/float(data_dict[key]['to_messages'])+float(data_dict[key]['from_messages']))
        
        
### removing the feature email_address since it is not handled currenlty in featureFormat
features_list.remove('email_address')


### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

target_names = ["Not POI", "POI"]


### First i would like to see the accuracy for alogrithm for all the features.

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf=clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
accuracy=clf.score(features_test,labels_test)


print("Accuracy for GaussianNB is ",accuracy)
pp.pprint (classification_report(y_true=labels_test, y_pred=pred, target_names=target_names))

### Accuracy using the support vector machine

from sklearn import svm
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(features_train)
pca.transform(features_train)
clf=svm.SVC()
clf.fit(features_train,labels_train)
pca.transform(features_test)
pred=clf.predict(features_test)
accuracy=clf.score(features_test,labels_test)

print("Accuracy for SVM is ",accuracy)
pp.pprint (classification_report(y_true=labels_test, y_pred=pred, target_names=target_names))

### lets use decision tree and check the classification

from sklearn import tree
clf=tree.DecisionTreeClassifier(min_samples_split=40)
clf=clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)

print("Accuracy for Decision Tree is ",accuracy)
pp.pprint (classification_report(y_true=labels_test, y_pred=pred, target_names=target_names))

### let knn classifier and check its accuracy
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3)
clf=clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)

print("Accuracy for Knn decision Tree is ",accuracy)
pp.pprint (classification_report(y_true=labels_test, y_pred=pred, target_names=target_names))


#### using Select K Best algo to selection the best 7 features after ty.

no_of_selected_feat=10
from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest(f_classif, k= no_of_selected_feat)
kbest.fit_transform(features, labels)
features_selected=[features_list[i+1] for i in kbest.get_support(indices=True)]
features_score={}
for a,b in zip(features_selected,kbest.scores_):
    features_score[a]=b

#### publishing the top 10 features from best score to fewer score

print ('\n')
print ("SelectKBest chose " + str(no_of_selected_feat) + " features")
pp.pprint(sorted(features_score.items(), key=itemgetter(1),reverse=True))
print ("\n")
    
### Adding poi feature at the begining fo the features list.

if 'poi' in features_selected:
    features_selected.remove('poi')
    features_selected.insert(0, 'poi')
    
else:
    features_selected.insert(0, 'poi')
    features_selected=features_selected[:-1]
    


### Adding newly engineered features to feature_list.

features_list = features_selected
features_list.append('bonus_to_tot_pay_ratio')
features_list.append('poi_to_total_email_ratio')

#### formating the features for the valid values.

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#### validating the data for top 10 features select from K Best algorithium.

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    

### Accuracy using the GaussianNB using best 12 features.

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf=clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
accuracy=clf.score(features_test,labels_test)

print("Accuracy for GaussianNB(using KBest) is ",accuracy)
pp.pprint (classification_report(y_true=labels_test, y_pred=pred, target_names=target_names))


### Accuracy using the support vector machine using best 12 features.

from sklearn import svm
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(features_train)
pca.transform(features_train)
clf=svm.SVC()
clf.fit(features_train,labels_train)
pca.transform(features_test)
pred=clf.predict(features_test)
accuracy=clf.score(features_test,labels_test)

print("Accuracy for SVM(using KBest) is ",accuracy)
pp.pprint (classification_report(y_true=labels_test, y_pred=pred, target_names=target_names))

### lets use decision tree and check the classification using best 12 features.

from sklearn import tree
clf=tree.DecisionTreeClassifier(min_samples_split=40)
clf=clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)

print("Accuracy for Decision Tree(using KBest) is ",accuracy)
pp.pprint (classification_report(y_true=labels_test, y_pred=pred, target_names=target_names))


### let knn classifier and check its accuracy using best 12 features.

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3)
clf=clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)

print("Accuracy for Knn decision Tree(using KBest) is ",accuracy)
pp.pprint (classification_report(y_true=labels_test, y_pred=pred, target_names=target_names))

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


### Creating a pipeline with scaling ,pca and GaussianNB algorithium.

gnb=GaussianNB()
scaler = StandardScaler()
pipe=Pipeline(steps=[('scaling',scaler),('pca',pca),('gnb',gnb)])
print (pipe.get_params().keys())

###parameter with differnent values of feature for n_component use values 3 and 5.

parameter={"pca__n_components": [3,5],
              "pca__random_state":[42]#,
               #"svm__kernel":['rbf','linear','poly'],
               #"svm__C":[10]
               
              #"knn__n_neighbors" :[3,5],
              #"knn__algorithm" :['auto','ball_tree','kd_tree','brute'],
              #"knn__weights":['uniform','distance'],
              #"knn__leaf_size":[40]
              }

### GridSearchCV allow to use the all the combination of provided featuers.

grid = GridSearchCV(pipe, parameter)
grid.fit(features, labels)
clf = grid.best_estimator_
pred = clf.predict(features_test)

print (classification_report(y_true=labels_test, y_pred=pred, target_names=target_names))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

### dump the classifier ,dataset and features_list.

dump_classifier_and_data(clf, my_dataset, features_list)

### testing classifier ,dataset and features_list for performance metrices.

test_classifier(clf, my_dataset, features_list)