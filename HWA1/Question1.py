import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
csvpath = "speed_dating_assignment.csv"
df = pd.read_csv(csvpath)
df
df.head()


###############QUESTION1############################

#filter relevant columns
df_filtered = df[['iid', 'age', 'pid', 'dec']]
#generate parter ages
df_partners = df_filtered.copy()
df_partners = df[['iid', 'age']]
df_partners.rename(index=str, columns={'iid': 'pid', 'age': 'partner_age'})
#df_partners.set_index(['pid', 'partner_age'], inplace=True)
df_partners = df_partners.drop_duplicates()
df2 = pd.merge(df_filtered, df_partners, left_on=['pid'], right_on = 'iid', how='left')

df2
#end of generating partner ages


pd.crosstab(df2['age_x'], df2['age_y'])#, margins=True)

df[df["gender"]==0].count()
###############QUESTION2############################
#Sample containing 80% of females
femaleSample = df[df["gender"]==0].sample(frac=0.8)

df.set_diff(femaleSample)

#Sample containing 80% of males
maleSample = df[df["gender"]==1].sample(frac=0.8)
