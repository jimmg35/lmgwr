import pandas as pd
import sys

df = pd.read_csv("./data/income_health.csv")
print(f"記憶體使用量: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
