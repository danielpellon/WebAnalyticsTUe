import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csvpath = "speed_dating_assignment.csv"
df = pd.read_csv(csvpath)

df.head()

df_filtered = df[['iid', 'age', 'pid', 'dec']]
df_partners = df_filtered.copy()
df_partners = df[['iid', 'age']]
df_partners.columns = ['pid', 'partner_age']
df_partners = df_partners.drop_duplicates()
df2 = pd.merge(df_filtered, df_partners, left_on=[
               'pid'], right_on='pid', how='left')
df2.head()
df2.describe()

plt.imshow(df2)
plt.show()


# Filter on positive decisions using a boolean mask
df2 = df2[df2['dec'] == 1]

matrix = pd.crosstab(df2['age'], df2['partner_age'], margins=True)
matrix
plt.subplots()
sns.heatmap(matrix)
plt.show()
