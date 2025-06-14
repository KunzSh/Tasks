import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/ (1+ np.exp(-z))
def computeloss(y,hx):
    return -np.mean(y* np.log(hx) +(1-y)* np.log(1-hx))
def computegradient(X,y,hx):
    return np.dot(X.T, (hx-y))/y.shape[0]
np.random.seed(42)
n_samples = 111

Xsize = np.random.normal(1500, 300, n_samples)  # feature 1: size & feature 2: rooms
Xrooms = np.random.randint(2, 6, n_samples)     
house_price = 50 * Xsize + 10000 * Xrooms + np.random.normal(0, 10000, n_samples)

median_price = np.median(house_price)
y = (house_price > median_price).astype(int) 

X = np.column_stack((Xsize, Xrooms))  

Xmean = X.mean(axis=0)
Xstd = X.std(axis=0)
Xnorm = (X - Xmean) / Xstd

X_bias = np.hstack((np.ones((Xnorm.shape[0],1)),Xnorm))

weights = np.zeros(X_bias.shape[1])  
learning_rate = 0.1
iterations = 1000
loss_history = []

for i in range(iterations):
      z = np.dot(X_bias, weights)
                          
hx = sigmoid(z)

loss = computeloss(y, hx)
grad = computegradient(X_bias, y, hx)
weights -= learning_rate * grad
loss_history.append(loss)
if i % 100 == 0:
    
        print(f"Iteration {i}: Loss = {loss:.4f}")

final_probs = sigmoid(np.dot(X_bias, weights))
predictions = (final_probs >= 0.5).astype(int)

accuracy = np.mean(predictions == y)
print(f"\nFinal Accuracy: {accuracy:.4f}")


print("\nSample predictions:")
for i in range(5):
    print(f"Predicted: {predictions[i]} | Actual: {y[i]}")

x1_vals = np.linspace(Xnorm[:, 0].min() - 1, Xnorm[:, 0].max() + 1, 100)
x2_vals = np.linspace(Xnorm[:, 1].min() - 1, Xnorm[:, 1].max() + 1, 100)
xx1, xx2 = np.meshgrid(x1_vals, x2_vals)

grid = np.c_[xx1.ravel(), xx2.ravel()]
gridwithbias = np.hstack((np.ones((grid.shape[0], 1)), grid))


grid_preds = sigmoid(np.dot(gridwithbias, weights)).reshape(xx1.shape)

plt.figure(figsize=(8, 6))
contour = plt.contourf(xx1, xx2, grid_preds, levels=[0, 0.5, 1], alpha=0.3, colors=["blue", "red"])
plt.colorbar(contour)

# Plot actual data points
plt.scatter(Xnorm[y == 0][:, 0], Xnorm[y == 0][:, 1], color='blue', label='Price ≤ median')
plt.scatter(Xnorm[y == 1][:, 0], Xnorm[y == 1][:, 1], color='red', label='Price > median')

plt.xlabel("Normalized Size (sqft)")
plt.ylabel("Normalized Rooms")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()
