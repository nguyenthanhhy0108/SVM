import numpy as np
import matplotlib.pyplot as plt

def generate_data(seed=2252):
    np.random.seed(seed)
    X_positive = np.random.normal(loc=[2, 3], scale=1, size=(50, 2))
    X_negative = np.random.normal(loc=[6, 7], scale=1, size=(50, 2))
    X = np.vstack([X_positive, X_negative])
    y = np.hstack([np.ones(50), -np.ones(50)])
    return X, y

def lagrange(alpha, X, y):
    return -np.sum(alpha) + 0.5 * np.sum(alpha * alpha * y * y * np.dot(X, X.T))

def gradient(alpha, X, y):
    return 1 - alpha * y * np.dot(X, X.T)

def gradient_descent(X, y, learning_rate=0.01, epochs=50):
    N, _ = X.shape
    alpha = np.zeros(N)
    objectives = []
    for epoch in range(epochs):
        grad = gradient(alpha, X, y)
        alpha = alpha + learning_rate * grad
        alpha = np.maximum(0, np.minimum(1, alpha))
        objectives.append(lagrange(alpha, X, y))
    return alpha, objectives

X, y = generate_data()

optimal_alpha, objectives = gradient_descent(X, y)



# plt.plot(objectives)
# plt.xlabel('Epochs')
# plt.ylabel('Lagrange Value')
# plt.title('Gradient Descent for SVM Lagrange Function')
# plt.savefig('Optimal Progress')
# plt.show()


# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k', label='Data points')
# plt.title('Synthetic Data')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.savefig('Data')
# plt.show()

a = optimal_alpha[-1].T
a = np.array(a).tolist()

print(a)

plt.plot(a, label='Data Line')

plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Line Plot')
plt.legend()

plt.savefig('Data')
plt.show()