
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(0)  
data = np.random.normal(loc=60, scale=15, size=100)  

mean_val = np.mean(data)
median_val = np.median(data)
mode_val = stats.mode(data, keepdims=False).mode  

print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"Mode: {mode_val:.2f}")


plt.figure(figsize=(8, 5))
plt.hist(data, bins=15, color='blue', edgecolor='black', alpha=0.7)
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_val:.2f}')
plt.axvline(mode_val, color='purple', linestyle='dashed', linewidth=1.5, label=f'Mode: {mode_val:.2f}')
plt.title('Histogram of Simulated Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



