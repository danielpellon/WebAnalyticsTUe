import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import tree
import numpy as np
import seaborn as sns
import math
import graphviz

csvpath = "speed_dating_assignment.csv"
df_import = pd.read_csv(csvpath)
# Reformat numbers to remove commas
df_import['zipcode'] = df_import['zipcode'].str.replace(',', '')
df_import['income'] = df_import['income'].str.replace(',', '')
df_import['mn_sat'] = df_import['mn_sat'].str.replace(',', '')
df_import['tuition'] = df_import['tuition'].str.replace(',', '')

# Build a dataframe that only includes the attributes known on beforehand
# Only select attributes that are known before the date
df_filtered = df_import.loc[:, :'dec']
# Remove irrelevant and string-based attributes
df_filtered.drop(['id', 'idg', 'partner', 'undergra', 'field', 'from', 'career', 'match'], axis=1, inplace=True)

# add partner age to the dataframe
df_partners = df_filtered.copy()
# df_partners = df_partners.filter(items=['iid', 'age'])
df_partners.columns = 'partner_' + df_partners.columns
df_partners.rename(columns={'partner_iid': 'pid'}, inplace=True)
df_partners.drop_duplicates(subset=['pid'], inplace=True)
df_partners.drop(['partner_dec'], axis=1, inplace=True)
df_both = pd.merge(df_filtered, df_partners, on=['pid'], how='left')

# pd.set_option('display.max_columns', None)
# df.head()
# df.describe()

###############QUESTION1.1############################
# generate age matrix
df_ageMatrix = df_both[['iid', 'age', 'pid', 'dec', 'partner_age']]

# Function to format cells for the heatmap
def format_cells(x):
    if math.isnan(x):
        return "—"
    return "{:.0%}".format(x)

# build a matrix with age (rows) and partner age (columns); decision rate is shown in cells
matrix = pd.pivot_table(df_ageMatrix, values='dec', index='age', columns='partner_age')

#Running this line for me throws an error, but matrix still works .daniel
cm = sns.light_palette("orange", as_cmap=True)
(matrix.style
 .background_gradient(cmap=cm)
 .format(format_cells)
 .highlight_null('white')
 )

###############QUESTION1.2############################
###################Females############################
# Sample containing 80% of females
femaleTrain = df_both[df_both['gender'] == 0].sample(frac=0.8)
femaleFilter = df_both[df_both['gender'] == 0].drop(femaleTrain.index)

# replace NaN with medians
femaleTrain.fillna(value=femaleTrain.median(), inplace=True)
femaleFilter.fillna(value=femaleFilter.median(), inplace=True)

# Create femaleTest: a dataframe for the test without the results
femaleTest = femaleFilter.drop(['dec'], axis=1)

# Creating the decision tree
target_f = femaleTrain['dec']
features_f = femaleTrain.drop(['dec'], axis=1).values

# Here is where we will controll things such as overfitting
test_tree_f = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=10)
test_tree_f = test_tree_f.fit(features_f, target_f)

# Run the tree
predictions_f = test_tree_f.predict(femaleTest)

# Evaluate the tree
sk.metrics.accuracy_score(femaleFilter['dec'], predictions_f, normalize=True)
sk.metrics.precision_score(femaleFilter['dec'], predictions_f)
sk.metrics.f1_score(femaleFilter['dec'], predictions_f)

# Visualise the tree
dot_data = tree.export_graphviz(test_tree_f, out_file=None, feature_names=femaleTrain.columns[1:], class_names=['Reject','Accept'])
graph = graphviz.Source(dot_data)
graph.render("test_tree_female")

###################Males##############################
# Sample containing 80% of males
maleTrain = df_both[df_both['gender'] == 1].sample(frac=0.8)
maleFilter = df_both[df_both['gender'] == 1].drop(maleTrain.index)

# replace NaN with medians
maleTrain.fillna(value=maleTrain.median(), inplace=True)
maleFilter.fillna(value=maleFilter.median(), inplace=True)

# Create maleTest: a dataframe for the test without the results
maleTest = maleFilter.drop(['dec'], axis=1)

# Creating the decision tree
target_m = maleTrain['dec']
features_m = maleTrain.drop(['dec'], axis=1).values

# Here is where we will controll things such as overfitting
# Playing with max_depth and min_samples_split will increase/decrease accuracy
test_tree_m = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=10)
test_tree_m = test_tree_m.fit(features_m, target_m)

# Run the tree
predictions_m = test_tree_m.predict(maleTest)

# Evaluate the tree
sk.metrics.accuracy_score(maleFilter['dec'], predictions_m, normalize=True)
sk.metrics.precision_score(maleFilter['dec'], predictions_m)
sk.metrics.f1_score(maleFilter['dec'], predictions_m)

# Visualise the tree
dot_data = tree.export_graphviz(test_tree_m, out_file=None, feature_names=maleTrain.columns[1:], class_names=['Reject','Accept'])
graph = graphviz.Source(dot_data)
graph.render("test_tree_male")

###############QUESTION1.3############################
df_score = df_both.copy()

#Calculates score by multiplying the partners score for a certain attribute with the importance of the attribute
df_score['attribute_score'] = df_score.partner_attr3_1 * df_score.attr1_1 + df_score.partner_sinc3_1 * df_score.sinc1_1 + df_score.partner_intel3_1 * df_score.intel1_1 + df_score.partner_fun3_1 * df_score.fun1_1 + df_score.partner_amb3_1 * df_score.amb1_1
###################Females############################
# Sample containing 80% of females
femaleTrain_score = df_score[df_score['gender'] == 0].sample(frac=0.8)
femaleFilter_score = df_score[df_score['gender'] == 0].drop(femaleTrain_score.index)

# replace NaN with medians
femaleTrain_score.fillna(femaleTrain_score.median(), inplace=True)
femaleFilter_score.fillna(femaleFilter_score.median(), inplace=True)

# Create femaleTest_score: a dataframe for the test without the results
femaleTest_score = femaleFilter_score.drop(['dec'], axis=1)

# Creating the decision tree
target_f_score = femaleTrain_score['dec']
features_f_score = femaleTrain_score.drop(['dec'], axis=1).values

# Here is where we will controll things such as overfitting
test_tree_f_score = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=10)
test_tree_f_score = test_tree_f_score.fit(features_f_score, target_f_score)

# Run the tree
predictions_f_score = test_tree_f_score.predict(femaleTest_score)

# Evaluate the tree
sk.metrics.accuracy_score(femaleFilter_score['dec'], predictions_f_score, normalize=True)
sk.metrics.precision_score(femaleFilter_score['dec'], predictions_f_score)
sk.metrics.f1_score(femaleFilter_score['dec'], predictions_f_score)

# Visualise the tree
dot_data = tree.export_graphviz(test_tree_f_score, out_file=None, feature_names=femaleTrain_score.columns[1:], class_names=['Reject','Accept'])
graph = graphviz.Source(dot_data)
graph.render("test_tree_female_score")

###################Males##############################
# Sample containing 80% of males
maleTrain_score = df_score[df_score['gender'] == 1].sample(frac=0.8)
maleFilter_score = df_score[df_score['gender'] == 1].drop(maleTrain_score.index)

# replace NaN with medians
maleTrain_score.fillna(maleTrain_score.median(), inplace=True)
maleFilter_score.fillna(maleFilter_score.median(), inplace=True)

# Create maleTest_score: a dataframe for the test without the results
maleTest_score = maleFilter_score.drop(['dec'], axis=1)

# Creating the decision tree
target_m_score = maleTrain_score['dec']
features_m_score = maleTrain_score.drop(['dec'], axis=1).values

# Here is where we will controll things such as overfitting
test_tree_m_score = tree.DecisionTreeClassifier(max_depth=2, min_samples_split=10)
test_tree_m_score = test_tree_f_score.fit(features_m_score, target_m_score)

# Run the tree
predictions_m_score = test_tree_m_score.predict(maleTest_score)

# Evaluate the tree
sk.metrics.accuracy_score(maleFilter_score['dec'], predictions_m_score, normalize=True)
sk.metrics.precision_score(maleFilter_score['dec'], predictions_m_score)
sk.metrics.f1_score(maleFilter_score['dec'], predictions_m_score)

# Visualise the tree
dot_data = tree.export_graphviz(test_tree_m_score, out_file=None, feature_names=maleTrain_score.columns[1:], class_names=['Reject','Accept'])
graph = graphviz.Source(dot_data)
graph.render("test_tree_male_score")
