import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import tree
import numpy as np
import seaborn as sns
import math

csvpath = "speed_dating_assignment.csv"
df_import = pd.read_csv(csvpath)
df_import.replace(to_replace=',', value='', inplace=True)
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
test_tree = tree.DecisionTreeClassifier()
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

# Create femaleTest: a dataframe for the test without the results
maleTest = maleFilter.drop(['dec'], axis=1)


# Creating the decision tree
target = maleTrain['dec']
features = maleTrain.drop(['dec'], axis=1).values

# Here is where we will controll things such as overfitting
test_tree = tree.DecisionTreeClassifier()
test_tree = test_tree.fit(features, target)

# Run the tree
predictions = test_tree.predict(maleTest)
# Evaluate the tree
sk.metrics.accuracy_score(maleFilter['dec'], predictions, normalize=True)
sk.metrics.precision_score(maleFilter['dec'], predictions)
sk.metrics.f1_score(maleFilter['dec'], predictions)
sk.metrics.classification_report(maleFilter['dec'], predictions)

import graphviz
dot_data = tree.export_graphviz(test_tree, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
