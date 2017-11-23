########################QUESTION2.1############################
#Unisex model
import pandas as pd

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
df.drop(['gender'], axis = 1)
#We create the test and train samples
train = df.sample(frac=0.8)
test = df.drop(train.index)


test.drop(['dec'], axis=1)

#Then we will the missing values with the median
