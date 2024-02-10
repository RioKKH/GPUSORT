#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# データを準備（pandasのデータフレームからリストへ）
df = pd.rea_csv("sort_512.csv", header=0)

sns.violinplot(data=df)

plt.xticks(ticks=range(len(df.columns)), labels=df.columns)
plt.show()
