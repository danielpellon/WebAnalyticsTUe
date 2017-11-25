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
# df_import.replace(to_replace=',', value='', inplace=True)
df_import['zipcode'] = df_import['zipcode'].str.replace(',', '')
df_import['income'] = df_import['income'].str.replace(',', '')
df_import['mn_sat'] = df_import['mn_sat'].str.replace(',', '')
df_import['tuition'] = df_import['tuition'].str.replace(',', '')

###############QUESTION1.1############################
# add partner age to the dataframe
df_partners = df_import.copy()
df_partners = df_partners.filter(items=['iid', 'age'])
df_partners.columns = ['pid', 'partner_age']
df_partners = df_partners.drop_duplicates()
df = pd.merge(df_import, df_partners, on=['pid'], how='left')

# generate age matrix
df_ageMatrix = df[['iid', 'age', 'pid', 'dec', 'partner_age']]


# onze eerste functie, Hoera structuur
def format_cells(x):
    if math.isnan(x):
        return "—"
    return "{:.0%}".format(x)


# build a matrix with age (rows) and partner age (columns); decision rate is shown in cells
# matrix = pd.crosstab(df_ageMatrix['age'], df_ageMatrix['partner_age'], values=df_ageMatrix['dec'], aggfunc=[matrixValue])
matrix = pd.pivot_table(df, values='dec', index='age', columns='partner_age')

# matrix
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
femaleTrain = df[df['gender'] == 0].sample(frac=0.8)
femaleFilter = df.drop(femaleTrain.index)
femaleTrain.drop(['field', 'undergra', 'from', 'career', 'match', 'career_c'], axis=1, inplace=True)
femaleFilter.drop(['field', 'undergra', 'from', 'career', 'match', 'career_c'], axis=1, inplace=True)

# replace NaN with medians
femaleTrain['age'].fillna(femaleTrain['age'].median(), inplace=True)
femaleTrain['round'].fillna(femaleTrain['round'].median(), inplace=True)
femaleFilter['age'].fillna(value=femaleFilter['age'].median(), inplace=True)
femaleFilter['round'].fillna(value=femaleFilter['round'].median(), inplace=True)
femaleTrain.fillna(0, inplace=True)
femaleFilter.fillna(0, inplace=True)

# Create femaleTest: a dataframe for the test without the results
femaleTest = femaleFilter.drop(['dec'], axis=1)

# Creating the decision tree
target = femaleTrain['dec']
features = femaleTrain.drop(['dec'], axis=1).values

# Here is where we will controll things such as overfitting
test_tree = tree.DecisionTreeClassifier(max_depth = 2, min_samples_split = 10)
test_tree = test_tree.fit(features, target)

# Run the tree
predictions = test_tree.predict(femaleTest)

# Evaluate the tree
sk.metrics.accuracy_score(femaleFilter['dec'], predictions, normalize=True)
sk.metrics.precision_score(femaleFilter['dec'], predictions)
sk.metrics.f1_score(femaleFilter['dec'], predictions)
sk.metrics.classification_report(femaleFilter['dec'], predictions)

###################Males##############################
# Sample containing 80% of males
maleTrain = df[df['gender'] == 1].sample(frac=0.8)
maleFilter = df.drop(maleTrain.index)
maleTrain.drop(['field', 'undergra', 'from', 'career', 'match'], axis=1, inplace=True)
maleFilter.drop(['field', 'undergra', 'from', 'career', 'match'], axis=1, inplace=True)

# replace NaN with medians
maleTrain['age'].fillna(maleTrain['age'].median(), inplace=True)
maleTrain['round'].fillna(maleTrain['round'].median(), inplace=True)
maleFilter['age'].fillna(value=maleFilter['age'].median(), inplace=True)
maleFilter['round'].fillna(value=maleFilter['round'].median(), inplace=True)
maleTrain.fillna(0, inplace=True)
maleFilter.fillna(0, inplace=True)

# Create maleTest: a dataframe for the test without the results
maleTest = maleFilter.drop(['dec'], axis=1)

# Creating the decision tree
target = maleTrain['dec']
features = maleTrain.drop(['dec'], axis=1).values

# Here is where we will controll things such as overfitting
# Playing with max_depth and min_samples_split will increase/decrease accuracy
test_tree = tree.DecisionTreeClassifier(max_depth = 2, min_samples_split = 10)
test_tree = test_tree.fit(features, target)

# Run the tree
predictions = test_tree.predict(maleTest)
# Evaluate the tree
sk.metrics.accuracy_score(maleFilter['dec'], predictions, normalize=True)
sk.metrics.precision_score(maleFilter['dec'], predictions)
sk.metrics.f1_score(maleFilter['dec'], predictions)
sk.metrics.classification_report(maleFilter['dec'], predictions)

###############QUESTION1.3############################
df_score = df.copy()

#Calculates score by multiplying the partners score for a certain attribute with the importance of the attribute
df_score['attribute_score'] = df_score.attr*df_score.attr1_1 + df_score.sinc*df_score.sinc1_1 + df_score.intel*df_score.intel1_1 + df_score.fun*df_score.fun1_1 + df_score.amb*df_score.amb1_1 + df_score.shar*df_score.shar1_1

###################Females############################
# Sample containing 80% of females
femaleTrain_score = df_score[df_score['gender'] == 0].sample(frac=0.8)
femaleFilter_score = df_score.drop(femaleTrain_score.index)
femaleTrain_score.drop(['field', 'undergra', 'from', 'career', 'match', 'career_c'], axis=1, inplace=True)
femaleFilter_score.drop(['field', 'undergra', 'from', 'career', 'match', 'career_c'], axis=1, inplace=True)

# replace NaN with medians
femaleTrain_score['age'].fillna(value=femaleTrain_score['age'].median(), inplace=True)
femaleTrain_score['round'].fillna(value=femaleTrain_score['round'].median(), inplace=True)
femaleFilter_score['age'].fillna(value=femaleFilter_score['age'].median(), inplace=True)
femaleFilter_score['round'].fillna(value=femaleFilter_score['round'].median(), inplace=True)
femaleTrain_score.fillna(0, inplace=True)
femaleFilter_score.fillna(0, inplace=True)

# Create femaleTest_score: a dataframe for the test without the results
femaleTest_score = femaleFilter_score.drop(['dec'], axis=1)

# Creating the decision tree
target_f_score = femaleTrain_score['dec']
features_f_score = femaleTrain_score.drop(['dec'], axis=1).values

# Here is where we will controll things such as overfitting
test_tree_f_score = tree.DecisionTreeClassifier(max_depth = 5, min_samples_split = 5, min_samples_leaf = 50)
test_tree_f_score = test_tree_f_score.fit(features_f_score, target_f_score)

# Run the tree
predictions_f_score = test_tree_f_score.predict(femaleTest_score)

# Evaluate the tree
sk.metrics.accuracy_score(femaleFilter_score['dec'], predictions_f_score, normalize=True)
sk.metrics.precision_score(femaleFilter_score['dec'], predictions_f_score)
sk.metrics.f1_score(femaleFilter_score['dec'], predictions_f_score)
sk.metrics.classification_report(femaleFilter_score['dec'], predictions_f_score)

###################Males##############################
# Sample containing 80% of males
maleTrain_score = df_score[df_score['gender'] == 1].sample(frac=0.8)
maleFilter_score = df_score.drop(maleTrain_score.index)
maleTrain_score.drop(['field', 'undergra', 'from', 'career', 'match', 'career_c'], axis=1, inplace=True)
maleFilter_score.drop(['field', 'undergra', 'from', 'career', 'match', 'career_c'], axis=1, inplace=True)

# replace NaN with medians
maleTrain_score['age'].fillna(value=maleTrain_score['age'].median(), inplace=True)
maleTrain_score['round'].fillna(value=maleTrain_score['round'].median(), inplace=True)
maleFilter_score['age'].fillna(value=maleFilter_score['age'].median(), inplace=True)
maleFilter_score['round'].fillna(value=maleFilter_score['round'].median(), inplace=True)
maleTrain_score.fillna(0, inplace=True)
maleFilter_score.fillna(0, inplace=True)

# Create maleTest_score: a dataframe for the test without the results
maleTest_score = maleFilter_score.drop(['dec'], axis=1)

# Creating the decision tree
target_m_score = maleTrain_score['dec']
features_m_score = maleTrain_score.drop(['dec'], axis=1).values

# Here is where we will controll things such as overfitting
test_tree_m_score = tree.DecisionTreeClassifier(max_depth = 5, min_samples_split = 5, min_samples_leaf = 50)
test_tree_m_score = test_tree_f_score.fit(features_m_score, target_m_score)

# Run the tree
predictions_m_score = test_tree_m_score.predict(maleTest_score)

# Evaluate the tree
sk.metrics.accuracy_score(maleFilter_score['dec'], predictions_m_score, normalize=True)
sk.metrics.precision_score(maleFilter_score['dec'], predictions_m_score)
sk.metrics.f1_score(maleFilter_score['dec'], predictions_m_score)
sk.metrics.classification_report(maleFilter_score['dec'], predictions_m_score)
