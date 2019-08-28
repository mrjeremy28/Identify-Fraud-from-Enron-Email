#!/usr/bin/python

import sys
from time import time
import pickle
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

# import pretty print and set indent to 4
import pprint
pp = pprint.PrettyPrinter(indent = 4)
# import numpy
import numpy as np

# pandas and MarkdownTableWriter library
import pandas as pd
from pytablewriter import MarkdownTableWriter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'fraction_to_poi', 'deferred_income', 'long_term_incentive', 'restricted_stock', 'total_payments', 'shared_receipt_with_poi'] # You will need to use more features
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 
        'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
        'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 
        # (all units are in US dollars)
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

# create total features with poi + financial + email
total_features = ['poi'] + financial_features + email_features

### Load the dictionary containing the dataset
with open("./final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Print total number of data points
print 'Total Number of data points: ', len(data_dict)

# find features with missing values
features_with_missing_values = {}
for key in total_features:
    features_with_missing_values[key] = 0

# find number of poi
poi_list = []
for employee_key in data_dict:
    # if employee is poi then insert into poi list
    if (data_dict[employee_key]['poi'] == True):
        poi_list.append(data_dict[employee_key])
        print "Person of Interest is: ", employee_key
        # print data_dict[employee_key].keys()
    # increment missing features key if NaN or blank
    for feature in data_dict[employee_key]:
        if data_dict[employee_key][feature] == 'NaN' or feature not in features_with_missing_values:
            features_with_missing_values[feature] += 1
            # data_dict[employee_key][feature] = 0


# print number of poi
print "Number of POI: ", len(poi_list)
# print number of non poi
print "Number of Non-POI: ", len(data_dict) - len(poi_list)

# print total number of features
print "Total Number of features: ", len(total_features)
print total_features



# Remove TOTAL as an outlier
data_dict.pop("TOTAL", 0)



### Task 3: Create new feature(s)
# There was a lesson that had messages from and to poi

### Store to my_dataset for easy export below.
my_dataset = data_dict

# email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 
#                  'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

# loop through employees and calculate fraction to/from poi
for employee in my_dataset:
    # calculate the fraction of emails from poi
    my_dataset[employee]["fraction_from_poi"] = 0.
    # divide from_poi_to_this_person by to_messages
    if my_dataset[employee]["from_poi_to_this_person"] != 'NaN' and my_dataset[employee]["to_messages"]!='NaN' and my_dataset[employee]["to_messages"] != 0:
        my_dataset[employee]["fraction_from_poi"] = my_dataset[employee]["from_poi_to_this_person"] / float(my_dataset[employee]["to_messages"])
    
    # calculate the fraction of emails to poi 
    my_dataset[employee]["fraction_to_poi"] = 0.
    # divide from_this_person_to_poi and from_messages
    if my_dataset[employee]["from_this_person_to_poi"] != 'NaN' and my_dataset[employee]["from_messages"]!='NaN' and my_dataset[employee]["from_messages"] != 0:
        my_dataset[employee]["fraction_to_poi"] = my_dataset[employee]["from_this_person_to_poi"] / float(my_dataset[employee]["from_messages"])
    # print employee
    # print my_dataset[employee]

# Create new feature list for selecting best feature
new_feature_list = total_features
new_feature_list += ["fraction_from_poi", "fraction_to_poi"]

# need to remove email from new_feature_list b/c of conversion to float error
new_feature_list.remove('email_address')
original_feature_list = total_features
# original_features = original_features.remove('email_address') # removed by new_feature_list.remove() reference

# print new feature list
print "Features list with 2 created features:"
print new_feature_list


"""
START HERE FOR KBESTScores
run_kbest('all') to get list of scores
for k in range(1:15):
    run_kbest()
    use scores to get features
    append poi to beggining
    run classifiers to get scores of each
    output k, precision, recall
"""
# new_feature_list = features_list
# Create function to choose the 2nd element for sorting later
def choose_2nd_element(element):
    return element[1]

def run_kbest(dataset, feature_list, kparam):
    """ This function takes a dataset and feature_list and kparam 
    and returns the scores for each feature
    """
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    # from sklearn.model_selection import train_test_split
    # features_train, features_test, labels_train, labels_test = \
    #     train_test_split(features, labels, test_size = 0.35, random_state = 42)

    from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression
    # Select features using KBest
    feature_select = SelectKBest(f_regression, k=kparam)
    # Train using features 
    feature_select.fit(features, labels)

    # print out scores to get a preview
    # print "KBest scores raw:"
    # print feature_select.scores_
    return feature_select.scores_


def mean_scores(clf, classifier_name, features, labels, iters = 80):
    """ given a classifier and features, labels, iterate through random
    state for the classifier and output the mean accuracy, precision and recall
    """
    acc = []
    pre = []
    recall = []
    t0 = time()
    # iteration based on parameter of function
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size = 0.35, random_state = i)
        clf.fit(features_train, labels_train)
        predicts = clf.predict(features_test)

        acc = acc + [accuracy_score(labels_test, predicts)] 
        pre = pre + [precision_score(labels_test, predicts)]
        recall = recall + [recall_score(labels_test, predicts)]
        
    # output time for testing algorithm performance
    print "{0} took a total time of {1}".format(classifier_name, round(time()-t0, 3), "s")
    # print "accuracy: {}".format(np.mean(acc))
    # print "precision: {}".format(np.mean(pre))
    # print "recall: {}".format(np.mean(recall))

    # retun mean accuracy, precision, and recall
    return np.mean(acc), np.mean(pre), np.mean(recall)
    
def algo_get_scores(clf, classifier_name, dataset, features, scale = True):
    from sklearn import preprocessing
    # create data from features and dataset
    data = featureFormat(dataset, features, sort_keys = True)
    # create labels and features
    labels, features = targetFeatureSplit(data)
    # if scale is True then Scale the data
    if scale:
        scaler = preprocessing.MinMaxScaler()
        features = scaler.fit_transform(features)

    # grab the scores for the classifier
    acc_score, prec_score, rec_score = mean_scores(clf, classifier_name, features, labels)
    # return values
    return acc_score, prec_score, rec_score


f_list = new_feature_list[1:]

kparams = range(1, len(new_feature_list))
k_score_list = []
for kparam in kparams:
    k_scores = run_kbest(my_dataset, new_feature_list, kparam)
    scores = zip(f_list, k_scores)
    scores = sorted(scores, key=choose_2nd_element, reverse = True)
    kBest_features = ['poi'] + [(i[0]) for i in scores[0:kparam]]
    # print len(kBest_features)
    # print kBest_features
    acc_score, prec_score, rec_score = algo_get_scores(LogisticRegression(tol=0.1, C=0.02, class_weight='balanced'), "Logistic Regression", my_dataset, kBest_features)
    k_score_list.append([kparam, prec_score, rec_score])
    # print acc_score, prec_score, rec_score 
    
# print k_score_list

'''   
# map features to scores, making sure to skip the first element which is poi
scores = zip(new_feature_list[1:], feature_select.scores_)
# Sort the scores using 2nd element which is the value, 
# sort in reverse to get highest values first
scores = sorted(scores, key=choose_2nd_element, reverse = True)
# print out scores
print "Scores sorted by highest first:"
pp.pprint(scores)
'''
# create kBest features by taking top 10 from scores list and add poi to begginning
# kBest_features = ['poi'] + [(i[0]) for i in scores[0:10]]
# print 'Top 10 KBest Features:', kBest_features

kbest_pd = pd.DataFrame(k_score_list, columns=['k', 'Precision', 'Recall'])
# kbest_pd = pd.DataFrame.from_dict(feature_dict, orient='index', columns=kparams)
# print kbest_pd
writer = MarkdownTableWriter()
# set table name
writer.table_name = "K Values at different values"
# create markdown table
writer.from_dataframe(
    kbest_pd
)
# display markdown table for copy/paste
writer.write_table()


'''
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn import tree
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=42, splitter='best')
clf = clf.fit(features, labels)
test_classifier(clf, my_dataset, new_feature_list)

###feature importances for feature selection process

feature_importances = clf.feature_importances_
indices = np.argsort(feature_importances)[::-1]
print("Feature ranking:")
for f in range(len(indices)):
    print("%d. %s (%f)" % (f + 1, new_feature_list[1:][indices[f]], feature_importances[indices[f]]))

def runKBest(kparam):
    from sklearn.feature_selection import SelectKBest
    k = kparam

    data = featureFormat(my_dataset, features_list)
    labels, features = targetFeatureSplit(data)
    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    d = dict(zip(features_list[1:], scores))
    sorted_pairs = [(i, d[i]) for i in sorted(d, key=d.get, reverse=True)]
    best_features = list(map(lambda x: x[0], sorted_pairs[:k]))
    print pp.pprint(sorted_pairs)
    print best_features

runKBest(6)
runKBest(10)
# runKBest('all')
'''