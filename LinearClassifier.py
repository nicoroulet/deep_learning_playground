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

  def compute_gradient(self, X, y, W, b):
    """ computes the gradient of W for the given set of samples
      X is the sample
      y is X's label
      W is the Linear Classifier's matrix
      b is the LC's bias
    """
    # Forward pass
    scores = W.dot(X) + b
    shifted_scores = scores - np.max(scores)
    # Commented lines enable to calculate the actual loss, but are not
    # required for the gradient
    # num = np.exp(shifted_scores[y])
    exp_scores = np.exp(shifted_scores)

    den = np.sum(exp_scores)#, axis=1, keepdims=True)
    # quot = num / den

    # loss = -np.log(quot)

    # Gradient
    dscores = exp_scores / den
    dscores[y] -= 1

    dW = dscores[:, np.newaxis].dot(X[np.newaxis, :])
    dW += W * self.r
    db = dscores

    return dW, db


class LinearClassifier:
  """ Linear classifier model. Uses a loss calculator """

  def __init__(self, n_classes, input_dim, loss_calculator=SoftmaxCalculator(),
               learning_rate=0.0003):
    # Values of W are sampled from a normal(0, sqrt(2/n)) distribution
    self.W = (np.random.randn(n_classes, input_dim) *
              np.sqrt(2 / (input_dim)))
    self.b = (np.random.randn(n_classes) *
              np.sqrt(2 / (input_dim)))
    self.loss_calculator = loss_calculator
    self.learning_rate = learning_rate

  def predict(self, sample):
    return self.W.dot(sample) + self.b

  def compute_loss(self, scores, correct_class):
    return self.loss_calculator.compute_loss(scores, correct_class, W=self.W)

  def measure_accuracy(self, dataset):
    accuracy = np.mean([float(np.argmax(self.predict(X)) == y)
                          for X, y in zip(dataset['test']['data'],
                                          dataset['test']['labels'])])
    return accuracy

  def measure_loss(self, dataset):
    loss = np.mean([self.compute_loss(self.predict(X), y)
                          for X, y in zip(dataset['test']['data'],
                                          dataset['test']['labels'])])
    return loss

  def train(self, dataset):
    Xs = dataset['train']['data']
    ys = dataset['train']['labels']
    for X, y in zip(Xs, ys):
      dW, db = self.loss_calculator.compute_gradient(X, y, self.W, self.b)
      self.W -= dW * self.learning_rate
      self.b -= db * self.learning_rate
