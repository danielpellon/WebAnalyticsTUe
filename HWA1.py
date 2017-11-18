import pandas as pd
import matplotlib.pyplot as plt

csvpath = "speed_dating_assignment.csv"
dataset = pd.read_csv(csvpath)

dataset.head()
plt.matshow(dataset.corr())
plt.show()
