import unittest as ut
import numpy as np

import LinearClassifier
import Dataset

class SoftmaxCalculatorTest(ut.TestCase):

  def test_compute_loss(self):
    scores = np.array([-2.85,0.86,0.28])
    correct_class = 2
    calculator = LinearClassifier.SoftmaxCalculator()
    self.assertAlmostEqual(calculator.compute_loss(scores, correct_class), 1.04,
                           places=2)

class LinearClassifierTest(ut.TestCase):
  
  def test_predict_matches_dimensions(self):
    lc = LinearClassifier.LinearClassifier(2, 5)
    self.assertEqual(lc.predict([1,2,3,4,5]).shape, (2,))


class DatasetTest(ut.TestCase):

  def test_load_cifar(self):
    cifar = Dataset.load_cifar()
    self.assertTrue('train' in cifar)
    self.assertTrue('test' in cifar)
    self.assertTrue('data' in cifar['train'])
    self.assertTrue('labels' in cifar['train'])
    self.assertTrue('data' in cifar['test'])
    self.assertTrue('labels' in cifar['test'])

if __name__ == '__main__':
  ut.main()