import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

import seaborn as sns
csvpath = "speed_dating_assignment.csv"
df_import = pd.read_csv(csvpath)
df_import
df_import.head()


###############QUESTION1.1############################

# add partner age to the dataframe
df_partners = df_import.copy()
df_partners = df_partners.filter(items=['iid', 'age'])
df_partners.columns = ['pid', 'partner_age']
df_partners = df_partners.drop_duplicates()
df = pd.merge(df_import, df_partners, on=[
    'pid'], how='left')

# filter relevant columns
df_filtered = df_import[['iid', 'age', 'pid', 'dec']]

# generate parter ages
#df_partners = df_filtered.copy()
#df_partners = df_import[['iid', 'age']]
#df_partners.columns = ['pid', 'partner_age']
#df_partners = df_partners.drop_duplicates()
df = pd.merge(df_import, df_partners, on=[
    'pid'], how='left')
df_ageMatrix = pd.merge(df_filtered, df_partners, on=[
    'pid'], how='left')
# end of generating partner ages

# build a matrix with age (rows) and partner age (columns); decision rate is shown in cells
matrix = pd.crosstab(df_ageMatrix['age'], df_ageMatrix['partner_age'],
                     values=df_ageMatrix['dec'], aggfunc=[np.mean])
# matrix

cm = sns.light_palette("orange", as_cmap=True)
(matrix.style
 .background_gradient(cmap=cm)
 .format("{:.0%}")
 .highlight_null('white')
 )

###############QUESTION1.2############################
# Sample containing 80% of females

femaleTrain = df[df['gender'] == 0].sample(frac=0.8)

femaleTest = df.drop(femaleTrain.index)

# Get rid of gender column
del femaleTrain["gender"]


# Sample containing 80% of males
maleTrain = df[df["gender"] == 1].sample(frac=0.8)

maleTest = df.drop(maleTrain.index)

del maleTrain["gender"]

# We start building the tree by defining target and predictor features
target = femaleTrain["dec"]

# femaleTest['partner_age']
femaleTrain
# femaleTrain['partner_age']
femaleTrain['age'].fillna(femaleTrain['age'].median(), inplace=True)
femaleTrain['round'].fillna(femaleTrain['age'].median(), inplace=True)


features = femaleTrain[["age", "round"]].values

femaleTrain.describe()
# Here is where we will controll things such as overfiting
test_tree = tree.DecisionTreeClassifier()

test_tree = test_tree.fit(features, target)

femaleTest['age'].dropna(inplace=True)
femaleTest['round'].dropna(inplace=True)

test_tree.predict(femaleTest)

femaleTest

#
