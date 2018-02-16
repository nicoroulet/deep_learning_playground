import numpy as np

class SoftmaxCalculator:
  """ Class omputes softmax loss and gradient """

  def __init__(self, r=0.01):
    self.r = r  # regularization coeficient

  def regularization_loss(self, W):
    return np.sum(np.square(W))

  def compute_loss(self, scores, correct_class, W=[]):
    # computing exp(scores[correct_class] + C) / sum(exp(scores + C))
    # where C = max(scores)
    shifted_scores = scores - np.max(scores)
    num = np.exp(shifted_scores[correct_class])
    den = np.sum(np.exp(shifted_scores))
    return -np.log(num / den) + self.regularization_loss(W) * self.r / 2

  def compute_gradient(self, X, y, W):
    # Forward pass
    X = np.concatenate((X, [1]))
    scores = W.dot(X)
    shifted_scores = scores - np.max(scores)
    # Commented lines enable to calculate the actual loss, but are not
    # required for the gradient
    # num = np.exp(shifted_scores[y])
    den = np.sum(np.exp(shifted_scores))
    # quot = num / den
    
    # loss = -np.log(quot)

    # Gradient
    dscores = np.exp(shifted_scores) / den
    dscores[y] -= 1
    dW = dscores[:, np.newaxis].dot(X[np.newaxis, :]) 
    dW[:,:-1] += W[:,:-1] * self.r

    return dW

# TODO: finish
"""
class SVMCalculator:

  def __init__(self, regularization=0.1):
    self.r = regularization

  def compute_loss(self, scores, correct_class, W=[]):
    correct_class_score = scores[correct_class]
    return (np.sum(np.maximum(0, scores - correct_class_score + 1)) - 1 + 
           self.r * np.sum(W))
"""

class LinearClassifier(object):
  """ Linear classifier model. Uses a loss calculator """

  def __init__(self, n_classes, input_dim, loss_calculator=SoftmaxCalculator(),
               learning_rate=0.0003):
    # Values of W are sampled from a normal(0, sqrt(2/n)) distribution
    self.W = (np.random.randn(n_classes, input_dim + 1) *
              np.sqrt(2 / (input_dim)))
    self.loss_calculator = loss_calculator
    self.learning_rate = learning_rate

  def predict(self, sample):
    return self.W.dot(np.concatenate((sample, [1])))

  def compute_loss(self, scores, correct_class):
    return self.loss_calculator.compute_loss(scores, correct_class, W=self.W)

  def measure_accuracy(self, dataset):
    accuracy = np.mean([float(np.argmax(self.predict(X/256)) == y)
                          for X, y in zip(dataset['test']['data'], 
                                          dataset['test']['labels'])])
    return accuracy

  def measure_loss(self, dataset):
    loss = np.mean([self.compute_loss(self.predict(X / 256), y)
                          for X, y in zip(dataset['test']['data'], 
                                          dataset['test']['labels'])])
    return loss


  def train(self, dataset):
    for X, y in zip(dataset['train']['data'], dataset['train']['labels']):
      dW = self.loss_calculator.compute_gradient(X/256, y, self.W)
      # Learn
      self.W -= dW * self.learning_rate
