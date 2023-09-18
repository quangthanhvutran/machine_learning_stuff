import numpy as np

class HomeMadeLinearRegression:
    def load_data(self, x, y):
        self.data = x
        self.data = y
    def gradient_descent(w_in, b_in, cost_function, compute_gradient, alpha, num_iters):
