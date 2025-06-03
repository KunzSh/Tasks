import numpy as np
from scipy import stats
np.random.seed(1)

data = np.random.randint(18, 61, size=15)

mean_val = np.mean(data)
median_val = np.median(data)
mode_val = stats.mode(data, keepdims=True)

print("Dataset (ages):", data)
print("Mean:", mean_val)
print("Median:", median_val)
print("Mode:", mode_val.mode[0], "(Count:", mode_val.count[0], ")")