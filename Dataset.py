import numpy as np
def load_cifar(cifar_folder='cifar-10'):
  def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
      cifar = pickle.load(fo, encoding='bytes')
    return cifar

  def merge(batches):
    return {
        'data' : np.concatenate([batch[b'data'] for batch in batches]),
        'labels' : np.concatenate([batch[b'labels'] for batch in batches])
    }

  batches = [unpickle('%s/data_batch_%d' % (cifar_folder,i)) for i in range(1, 6)]
  train = merge(batches)
  # merge is used to purge the keys here
  test = merge([unpickle('%s/test_batch' % cifar_folder)])  
  return {
      'train' : train,
      'test' : test
  }
