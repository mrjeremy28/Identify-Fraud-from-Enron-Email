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
with open("final_project_dataset.pkl", "r") as data_file:
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
# print total_features

# print missing value count of features
pp.pprint(features_with_missing_values)


### Task 2: Remove outliers

# create function to easily show scatter plot with 2 features
def show_scatter(data, x, y):
    # import matplot lib and show inline if possible
    import matplotlib.pyplot as plot
    # for jupyter notebook
    # % matplotlib inline
    
    # format features to get nice data to plot
    data_formatted = featureFormat(data, [x, y], sort_keys=True)
    
    # loop through points and plot as scatter
    for point in data_formatted:
        x_data = point[0]
        y_data = point[1]
        plot.scatter(x_data, y_data)
        
    # set labels
    plot.xlabel(x)
    plot.ylabel(y)
    # show plot
    plot.show()

# Function to show largest amount of feature in dataset for outlier
def show_possible_outlier(data, feature):
    largest_amount = 0
    # loop through keys and find largest amount and not NaN
    for key in data:
        if data[key][feature] != 'NaN':
            if data[key][feature] > largest_amount:
                largest_amount = data_dict[key][feature] 
                largest_amount_key = key
    # print largest amount
    print "Possible Outlier: "
    print largest_amount_key
    print largest_amount

# Show scatter for Bonus and salary
show_scatter(data_dict, "salary", "bonus")

# Show text of outiler
show_possible_outlier(data_dict, "salary")
show_possible_outlier(data_dict, "bonus")

# Remove TOTAL as an outlier
data_dict.pop("TOTAL", 0)

# Show scatter to validate removed
show_scatter(data_dict, "salary", "bonus")

# Show text if any more outlier
show_possible_outlier(data_dict, "salary")
show_possible_outlier(data_dict, "bonus")



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

# plot scatter of new features
show_scatter(my_dataset, "fraction_from_poi", "fraction_to_poi")

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

### Extract features and labels from dataset for local testing
# featureFormat removes features with all zeros and converts NaNs to 0
# data = featureFormat(my_dataset, new_feature_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)
from sklearn import preprocessing
data = featureFormat(my_dataset, new_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


# Pick out best feature using SelectKBest 
"""
params for score_func
f_classif - ANOVA F-value between label/feature for classification tasks.
mutual_info_classif - Mutual information for a discrete target.
chi2 - Chi-squared stats of non-negative features for classification tasks.
f_regression - F-value between label/feature for regression tasks.
mutual_info_regression - Mutual information for a continuous target.
SelectPercentile - Select features based on percentile of the highest scores.
SelectFpr - Select features based on a false positive rate test.
SelectFdr - Select features based on an estimated false discovery rate.
SelectFwe - Select features based on family-wise error rate.
GenericUnivariateSelect - Univariate feature selector with configurable mode.
"""
from sklearn.feature_selection import SelectKBest, f_classif
# Select features using KBest
feature_select = SelectKBest(f_classif, k=10)
# Train using features from targetFeatureSplit function
feature_select.fit(features, labels)

# print out scores to get a preview
print "KBest scores raw:"
print feature_select.scores_

# Create function to choose the 2nd element for sorting later
def choose_2nd_element(element):
    return element[1]

# map features to scores, making sure to skip the first element which is poi
scores = zip(new_feature_list[1:], feature_select.scores_)
# Sort the scores using 2nd element which is the value, 
# sort in reverse to get highest values first
scores = sorted(scores, key=choose_2nd_element, reverse = True)
# print out scores
print "Scores sorted by highest first:"
pp.pprint(scores)

# create kBest features by taking top 10 from scores list and add poi to begginning
kBest_features = ['poi'] + [(i[0]) for i in scores[0:10]]
print 'Top 10 KBest Features:', kBest_features



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# lesson 15 for evaluate_poi_identifier.py
# split data to 35% for training
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.35, random_state=42)

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

def algo_performance(clf, classifier_name, dataset, orig_features, new_features):
    # get scores for classifier using original features
    orig_accuracy_score, orig_precision_score, orig_recall_score = \
        algo_get_scores(clf, classifier_name, dataset, orig_features, True)
    
    # get scores for classifier using new features
    new_accuracy_score, new_precision_score, new_recall_score = \
        algo_get_scores(clf, classifier_name, dataset, new_features, True)
    
    # return dictionary 
    return {classifier_name: [orig_accuracy_score, orig_precision_score, orig_recall_score, new_accuracy_score, new_precision_score, new_recall_score]}

# set original and new features for performance comparing TESTING ONLY
# original_features = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income', 'long_term_incentive', 'restricted_stock', 'total_payments', 'shared_receipt_with_poi']
# new_features = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'fraction_to_poi', 'deferred_income', 'long_term_incentive', 'restricted_stock', 'total_payments', 'shared_receipt_with_poi']
# print original_features
# print new_features

# Import all algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm

def displayMarkdownFeatures(feature_list1, feature_list2, columnName1, columnName2):

    # create empty dictionary
    classifier_dict = {}
    # run perfomance for Naive Bayes
    naive = algo_performance(GaussianNB(), "Naive Bayes", my_dataset, feature_list1, feature_list2)
    classifier_dict.update(naive)
    # run perfomance for Decision Tree
    dt_tree = algo_performance(tree.DecisionTreeClassifier(), "Decision Tree", my_dataset, feature_list1, feature_list2)
    classifier_dict.update(dt_tree)
    # run perfomance for Random Forest
    rnd_forest = algo_performance(RandomForestClassifier(), "Random Forest", my_dataset, feature_list1, feature_list2)
    classifier_dict.update(rnd_forest)
    # run perfomance for AdaBoost
    ada_boost = algo_performance(AdaBoostClassifier(), "AdaBoost", my_dataset, feature_list1, feature_list2)
    classifier_dict.update(ada_boost)
    # run perfomance for Support Vector
    svm_svc = algo_performance(svm.SVC(), "Support Vector", my_dataset, feature_list1, feature_list2)
    classifier_dict.update(svm_svc)
    # print classifer_dict
    # print(classifier_dict)


    # create panda frame from dictionary, set column headers
    new_pd = pd.DataFrame.from_dict(classifier_dict, orient='index', columns=[columnName1 + ' features accuracy', 
                                                            columnName1 + ' features precision', 
                                                            columnName1 + ' features recall', 
                                                            columnName2 + ' features accuracy', 
                                                            columnName2 + ' features precision', 
                                                            columnName2 + ' features recall'])

    # using library for Markdown Table Writer
    writer = MarkdownTableWriter()
    # set table name
    writer.table_name = "Mean Accuracy, Precision and Recall for Features"
    # create markdown table
    writer.from_dataframe(
        new_pd,
        add_index_column=True
    )
    # display markdown table for copy/paste
    writer.write_table()


# Show table of original features vs new features added
displayMarkdownFeatures(original_feature_list, new_feature_list, 'original', 'new')

# Provided to give you a starting point. Try a variety of classifiers.
# Naive Bayes from nb_author_id.py
# from sklearn.naive_bayes import GaussianNB
# # set classifier
# clf_naive = GaussianNB()
# # reset timeer for logging time to train
# t0 = time()
# # fit features and labels
# clf_naive.fit(features_train, labels_train)
# # print training time
# print "Naive Bayes training time:", round(time()-t0, 3), "s"
# # reset timeer for logging time to predict
# t0 = time()
# # make prediction
# pred_naive = clf_naive.predict(features_test)
# # print time to predict
# print "Naive Bayes predicting time:", round(time()-t0, 3), "s"
# # Print Accuracy, Precision, and Recall stats
# print "Naive Bayes accuracy: ", accuracy_score(labels_test, pred_naive)
# print "Naive Bayes precision: ", precision_score(labels_test, pred_naive)
# print "Naive Bayes recall: ", recall_score(labels_test, pred_naive)

# # Decision Tree from dt_author_id.py
# from sklearn import tree
# # set classifier
# clf_tree = tree.DecisionTreeClassifier(min_samples_split = 50)
# # reset timeer for logging time to train
# t0 = time()
# # fit features and labels
# clf_tree.fit(features_train, labels_train)
# # print training time
# print "Decision Tree training time:", round(time()-t0, 3), "s"
# # reset timeer for logging time to predict
# t0 = time()
# # make prediction
# pred_tree = clf_tree.predict(features_test)
# # print time to predict
# print "Decision Tree predicting time:", round(time()-t0, 3), "s"
# # Print Accuracy, Precision, and Recall stats
# print "Decision Tree accuracy: ", accuracy_score(labels_test, pred_tree)
# print "Decision Tree precision: ", precision_score(labels_test, pred_tree)
# print "Decision Tree recall: ", recall_score(labels_test, pred_tree)

# # Random Forest Classifier from choose_your_own
# from sklearn.ensemble import RandomForestClassifier
# # set classifier
# clf_random_forest = RandomForestClassifier(n_estimators=10, min_samples_split=2)
# # reset timeer for logging time to train
# t0 = time()
# # fit features and labels
# clf_random_forest.fit(features_train, labels_train)
# # print training time
# print "Random Forest training time:", round(time()-t0, 3), "s"
# # reset timeer for logging time to predict
# t0 = time()
# # make prediction
# pred_random_forest = clf_random_forest.predict(features_test)
# # print time to predict
# print "Random Forest predicting time:", round(time()-t0, 3), "s"
# # Print Accuracy, Precision, and Recall stats
# print "Random Forest accuracy: ", accuracy_score(labels_test, pred_random_forest)
# print "Random Forest precision: ", precision_score(labels_test, pred_random_forest)
# print "Random Forest recall: ", recall_score(labels_test, pred_random_forest)

# # AdaBoost Classifier from choose_your_own
# from sklearn.ensemble import AdaBoostClassifier
# # set classifier
# clf_adaboost = AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
# # reset timeer for logging time to train
# t0 = time()
# # fit features and labels
# clf_adaboost.fit(features_train, labels_train)
# # print training time
# print "AdaBoost training time:", round(time()-t0, 3), "s"
# # reset timeer for logging time to predict
# t0 = time()
# # make prediction
# pred_adaboost = clf_adaboost.predict(features_test)
# # print time to predict
# print "AdaBoost predicting time:", round(time()-t0, 3), "s"
# # Print Accuracy, Precision, and Recall stats
# print "AdaBoost accuracy: ", accuracy_score(labels_test, pred_adaboost)
# print "AdaBoost precision: ", precision_score(labels_test, pred_adaboost)
# print "AdaBoost recall: ", recall_score(labels_test, pred_adaboost)

# # Support Vector Classification from svm_author_id.py
# from sklearn import svm
# # set classifier
# clf_logistic_regression = svm.SVC(gamma='auto')
# # reset timeer for logging time to train
# t0 = time()
# # fit features and labels
# clf_logistic_regression.fit(features_train, labels_train)
# # print training time
# print "Support Vector training time:", round(time()-t0, 3), "s"
# # reset timeer for logging time to predict
# t0 = time()
# # make prediction
# pred_logistic_regression = clf_logistic_regression.predict(features_test)
# # print time to predict
# print "Support Vector time:", round(time()-t0, 3), "s"
# # Print Accuracy, Precision, and Recall stats
# print "Support Vector accuracy: ", accuracy_score(labels_test, pred_logistic_regression)
# print "Support Vector precision: ", precision_score(labels_test, pred_logistic_regression)
# print "Support Vector recall: ", recall_score(labels_test, pred_logistic_regression)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
# split data for GridSearchCV
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.35, random_state=42)

def fine_tune_algo(clf, classifier_name, params, dataset, feature_list):
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn import preprocessing
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    scaler = preprocessing.MinMaxScaler()
    features = scaler.fit_transform(features)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.35, random_state=42)

    # run GridSearchCV with estimator set to chosen classifier
    clf_tune = GridSearchCV(estimator = clf, param_grid = params, cv=5, iid= True).fit(features, labels)
    
    # print out best Estimator params
    print "{0}: Best Estimator params: {1}".format(classifier_name, clf_tune.best_estimator_)
    return clf_tune.best_estimator_

# set parameters to iterate through
# default parameters for GaussianNB() is 
# 'priors' : ['None'],
# 'var_smoothing': 1e-09
params = {
}
# run GridSearchCV with estimator set to chosen classifier
clf_tune = GridSearchCV(estimator = GaussianNB(), param_grid = params, cv=5, iid= True).fit(features, labels)
# print out best Estimator params
print "Best Estiamtor: ", clf_tune.best_estimator_

# set params want to tune
dt_params = {
    'criterion':('gini', 'entropy'),
    'min_samples_split' : range(2,50),
    'splitter':('best','random')
}
# get params for best DT Performance
dt_best_params = fine_tune_algo(tree.DecisionTreeClassifier(), "Decision Tree", dt_params, my_dataset, kBest_features)

# print params to console
print dt_best_params
# create dictionary of algo perfomrance with Best paramaeters
dt_tree = algo_performance(dt_best_params, "Decision Tree", my_dataset, original_feature_list, new_feature_list)

# print scores for DT Tree with tuned parameters
print dt_tree

# run final accuracy score on kBest_features vs original features
displayMarkdownFeatures(original_feature_list, kBest_features, 'original', 'Best')

# make final classifier selection with Best Estiamtor params
clf = GaussianNB(priors=None, var_smoothing=1e-09)
# make final features selection
features_list = kBest_features


'''
from sklearn.model_selection import StratifiedShuffleSplit
def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    # cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    cv = StratifiedShuffleSplit(n_splits=folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    # for train_idx, test_idx in cv: 
    for train_idx, test_idx in cv.split(features, labels):    
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print ("Warning: Found a predicted label not == 0 or 1.")
                print ("All predictions should take value 0 or 1.")
                print ("Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print (clf)
        print (PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print (RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print ("")
    except:
        print ("Got a divide by zero when trying out:", clf)
        print ("Precision or recall may be undefined due to a lack of true positive predicitons.")
'''


# Test dataset with tester.py -> test_classifier
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)