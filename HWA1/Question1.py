import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
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

# generate age matrix
df_ageMatrix = df[['iid', 'age', 'pid', 'dec', 'partner_age']]

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
femaleFilter = df.drop(femaleTrain.index)

# Sample containing 80% of males
maleTrain = df[df["gender"] == 1].sample(frac=0.8)
maleTest = df.drop(maleTrain.index)

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

testForTree = femaleTest.filter(items=['age', 'round'])

testForTree['age'].fillna(value=20, inplace=True)
testForTree['round'].fillna(value=20, inplace=True)

testForTree['age'].median()

testForTree

testForTree.isna().values.any()
predictions = test_tree.predict(testForTree)

sk.metrics.accuracy_score(femaleTest['dec'], predictions, normalize=True)

comparison = femaleTest['dec']


predictions
