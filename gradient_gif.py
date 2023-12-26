import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

def gradient_descent(X, y, learning_rate=0.01, epochs=100):
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

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlabel('Epochs')
ax.set_ylabel('Lagrange Value')
ax.set_title('Gradient Descent for SVM Lagrange Function')
ax.set_xlim(0, len(objectives))
ax.set_ylim(min(objectives), max(objectives) + 10)

def update(frame):
    line.set_data(range(frame + 1), objectives[:frame + 1])
    return line,

animation = FuncAnimation(fig, update, frames=len(objectives), blit=True)
animation.save('Optimal_Progress.gif', writer='imagemagick', fps=30)
