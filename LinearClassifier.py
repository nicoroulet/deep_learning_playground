import numpy as np

import Helpers

def random_weights(dims, normalization):
    # Values are sampled from a normal(0, sqrt(2/n)) distribution
  return (np.random.randn(*dims) *
              np.sqrt(2 / (normalization)))

class Softmax:
  """ Class computes softmax loss and gradient """

  def regularization_loss(self, W):
    return np.sum(np.square(W))

  def compute_loss(self, scores, correct_class, W=[]):
    # computing exp(scores[correct_class] + C) / sum(exp(scores + C))
    # where C = max(scores)
    shifted_scores = scores - np.max(scores)
    num = np.exp(shifted_scores[correct_class])
    den = np.sum(np.exp(shifted_scores))
    return -np.log(num / den) + self.regularization_loss(W) * self.r / 2

  def compute_dscores(self, scores, correct_class):
    shifted_scores = scores - np.max(scores)
    exp_scores = np.exp(shifted_scores)
    den = np.sum(exp_scores)
    dscores = exp_scores / den
    dscores[correct_class] -= 1
    return dscores


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


class Classifier:

  def __init__(self, loss_calculator):
    self.loss_calculator = loss_calculator

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



class LinearClassifier(Classifier):
  """ Linear classifier model. Uses a loss calculator """

  def __init__(self, n_classes, input_dim, loss_calculator=Softmax(),
               learning_rate=0.0003, r=0.01):
    super(LinearClassifier, self).__init__(loss_calculator)
    self.W = random_weights((n_classes, input_dim), input_dim)
    self.b = random_weights((n_classes,), input_dim)
    self.learning_rate = learning_rate
    self.r = r  # regularization coeficient

  def predict(self, sample):
    return self.W.dot(sample) + self.b

  def compute_loss(self, scores, correct_class):
    return self.loss_calculator.compute_loss(scores, correct_class, W=self.W)

  def update_parameters(self, dscores, X):
    """ Updates parameters W and b with respect to a given score gradient.

    dscores is an array of derivatives of loss wrt each class score.
    X is the input that originated that gradient.
    """
    dW = dscores[:, np.newaxis].dot(X[np.newaxis, :])
    dW += self.W * self.r
    db = dscores

    self.W -= dW * self.learning_rate
    self.b -= db * self.learning_rate

  def train(self, dataset):
    Xs = dataset['train']['data']
    ys = dataset['train']['labels']
    for X, y in zip(Xs, ys):
      dscores = self.loss_calculator.compute_dscores(self.predict(X), y)
      self.update_parameters(dscores, X)


class ReLU:

  def __init__(self, layer):
    self.layer = layer
    self.last_output = None

  def evaluate(self, X):
    self.last_output = np.maximum(0, self.layer.evaluate(X))
    return self.last_output

  def backpropagate(self, dscores):
    dscores[self.last_output <= 0] = 0  # TODO: this should be redundant, check
    return self.layer.backpropagate(dscores)


class Layer:

  def __init__(self, in_dim, out_dim, learning_rate=.0003, r=0.001):
    self.W = random_weights((out_dim, in_dim), in_dim)
    self.b = random_weights((out_dim,), in_dim)
    self.last_input = None
    self.learning_rate = learning_rate
    self.r = r

  def evaluate(self, X):
    self.last_input = X
    return self.W.dot(X) + self.b

  def backpropagate(self, dscores):
    """ Updates parameters W and b with respect to a given score gradient.

    dscores is an array of derivatives of loss wrt each class score.
    X is the input that originated that gradient.

    Returns the gradient of the previous layer
    """
    dW = dscores[:, np.newaxis].dot(self.last_input[np.newaxis, :])
    dW += self.W * self.r
    db = dscores

    dprev_scores = self.W.T.dot(dscores)

    self.W -= dW * self.learning_rate
    self.b -= db * self.learning_rate

    return dprev_scores


class NeuralNetwork(Classifier):

  def __init__(self, n_classes, input_dim, inner_layer_sizes,
               loss_calculator=Softmax(), learning_rate=.001, r=0.01):
    sizes = [input_dim] + inner_layer_sizes + [n_classes]
    # print(sizes)
    self.layers = ([ReLU(Layer(in_dim, out_dim, r))
                      for in_dim, out_dim in zip(sizes, sizes[1:-1])] +
                    [Layer(*sizes[-2:], learning_rate, r)]
                  )
    self.loss_calculator = loss_calculator
    # self.layers = [random_weights((out_dim, in_dim), in_dim),
    #                random_weights((out_dim,), in_dim)
    #                   for in_dim, out_dim in zip(sizes, sizes[1:])]

  def predict(self, sample):
    for layer in self.layers:
      sample = layer.evaluate(sample)
    return sample

  # TODO: fix regularization
  def compute_loss(self, scores, correct_class):
    return self.loss_calculator.compute_loss(scores, correct_class)

  def update_parameters(self, dscores):
    """ Updates layers with respect to a given score gradient.

    dscores is an array of derivatives of loss wrt each class score.
    X is the input that originated that gradient.
    """
    for layer in reversed(self.layers):
      dscores = layer.backpropagate(dscores)

  def train(self, dataset):
    Xs = dataset['train']['data']
    ys = dataset['train']['labels']
    pbar = Helpers.ProgressBar()
    n_samples = Xs.shape[0]
    for i, (X, y) in enumerate(zip(Xs, ys)):
      pbar.update(i, n_samples)
      dscores = self.loss_calculator.compute_dscores(self.predict(X), y)
      self.update_parameters(dscores)
