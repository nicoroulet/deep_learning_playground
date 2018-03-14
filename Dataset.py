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

  def normalize(batch):
    data = batch['data'].astype('float64')
    data = (data - np.mean(data)) / np.std(data)
    batch['data'] = data
    return batch

  batches = [unpickle('%s/data_batch_%d' % (cifar_folder,i))
              for i in range(1, 6)]
  train = normalize(merge(batches))
  # merge is used to purge the keys here
  test = normalize(merge([unpickle('%s/test_batch' % cifar_folder)]))
  return {
      'train' : train,
      'test' : test
  }
