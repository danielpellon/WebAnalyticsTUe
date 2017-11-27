########################QUESTION2.1############################
#Unisex model
import pandas as pd
from sklearn import tree
import sklearn as sk

csvpath = "speed_dating_assignment.csv"
df_import = pd.read_csv(csvpath)
# df_import.replace(to_replace=',', value='', inplace=True)
df_import['zipcode'] = df_import['zipcode'].str.replace(',', '')
df_import['income'] = df_import['income'].str.replace(',', '')
df_import['mn_sat'] = df_import['mn_sat'].str.replace(',', '')
df_import['tuition'] = df_import['tuition'].str.replace(',', '')


#We compute the partner variable id and to the dataframe in the
#same way as before, we also drop the gender
df_partners = df_import.copy()
df_partners = df_partners.filter(items=['iid', 'age'])
df_partners.columns = ['pid', 'partner_age']
df_partners = df_partners.drop_duplicates()
df = pd.merge(df_import, df_partners, on=['pid'], how='left')
df.drop(['gender', 'field', 'undergra', 'from', 'career', 'match', 'career_c'], axis = 1, inplace= True)

#drop catergorical
#df.drop(['partner_field', 'partner_undergra', 'partner_from', 'partner_career', 'partner_match', 'partner_career_c', 'partner_dec'], axis=1, inplace=True)


#We create the test and train samples
train = df.sample(frac=0.8)
test = df.drop(train.index)


#Get rid of target variable in test
target_results = test['dec'].copy()

test.drop(['dec'], axis=1, inplace=True)


#Then we will the missing values with the median on both train and test_tree
train['age'].fillna(value=train['age'].median(), inplace=True)
train['round'].fillna(value=train['round'].median(), inplace=True)

test['age'].fillna(value=train['age'].median(), inplace=True)
test['round'].fillna(value=train['round'].median(), inplace=True)

#Fill missing values with median
train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)

###############Create decision tree##########################
#Chose target and features
target = train['dec']
features = train.drop(['dec'], axis=1).values

#Create tree
tree = tree.DecisionTreeClassifier(max_depth=2, min_samples_split=10);
#fit tree
tree = tree.fit(features, target)

train



#showtime

predictions = tree.predict(test)

sk.metrics.accuracy_score(target_results, predictions, normalize=True)
sk.metrics.precision_score(target_results, predictions)
sk.metrics.f1_score(target_results, predictions)
sk.metrics.classification_report(target_results, predictions)
