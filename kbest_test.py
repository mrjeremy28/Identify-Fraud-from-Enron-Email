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
"""
new_feature_list = features_list
# Create function to choose the 2nd element for sorting later
def choose_2nd_element(element):
    return element[1]

def run_kbest(dataset, feature_list, kparam):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    # from sklearn import preprocessing
    # scaler = preprocessing.MinMaxScaler()
    # features = scaler.fit_transform(features)


    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size = 0.35, random_state = 42)

    from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression
    # Select features using KBest
    feature_select = SelectKBest(f_regression, k=kparam)
    # Train using features from targetFeatureSplit function
    feature_select.fit(features_train, labels_train)

    # print out scores to get a preview
    print "KBest scores raw:"
    print feature_select.scores_
    return feature_select.scores_

f_list = new_feature_list[1:]
feature_dict = {k:[] for k in f_list}
kparams_columns = ['k=10', 'k=2', 'k=3', 'k=4', 'k=8', 'all']
kparams = [10, 2, 3, 4, 8, 'all']

for kparam in kparams:
    k_scores = run_kbest(my_dataset, new_feature_list, kparam)
    for i in range(0,len(f_list)):
        list_key = f_list[i]
        if list_key in feature_dict.keys():
            feature_dict[list_key].append(k_scores[i])


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

# feature_dict = dict(zip(new_feature_list[1:], []))


# kparams = ['k=10', 'k=2', 'k=3', 'k=4', 'k=8', 'all']
# kparams = ['k=10']
kbest_pd = pd.DataFrame.from_dict(feature_dict, orient='index', columns=kparams)
print kbest_pd
writer = MarkdownTableWriter()
# set table name
writer.table_name = "ss"
# create markdown table
writer.from_dataframe(
    kbest_pd,
    add_index_column=True
)
# display markdown table for copy/paste
writer.write_table()


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

data_df = pd.DataFrame.from_dict(data_dict, orient='index')
data_df.shape
data_df.replace(to_replace='NaN', value=np.nan, inplace=True)
data_df.count().sort_values()

data_df = data_df.drop(["email_address"], axis=1)
data_df = data_df.drop(["LOCKHART EUGENE E"], axis=0)
cols = [
    'poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income',
    'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees',
    'total_payments', 'exercised_stock_options', 'restricted_stock',
    'restricted_stock_deferred', 'total_stock_value',
    'from_poi_to_this_person', 'shared_receipt_with_poi', 'to_messages',
    'from_this_person_to_poi', 'from_messages'
]
data_df = data_df[cols]
# data_df.drop("TOTAL", inplace=True)

def do_split(data):
    X = data.copy()
    #Removing the poi labels and put them in a separate array, transforming it
    #from True / False to 0 / 1
    y = X.pop("poi").astype(int)
    
    return X, y, 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedShuffleSplit

pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', PCA(random_state=42)),
    ('classify', GaussianNB())
])

N_FEATURES_OPTIONS = [2, 4, 8]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS
    },
]
reducer_labels = ['PCA', 'KBest']
cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(pipe, cv=cv, n_jobs=1, param_grid=param_grid)
# digits = load_digits()
X, y = do_split(data_df)
grid.fit(X, y)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape( -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Digit classification accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')

plt.show()