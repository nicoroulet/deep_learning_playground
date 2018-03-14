import sys

class ProgressBar:

  def __init__(self, size=20):
    self.size = size
    sys.stdout.write('[%s]' % (' ' * self.size))

  def update(self, progress, total):
    scaled_progress = (progress * self.size) // total
    sys.stdout.write('\r[%s%s]' % ('-' * scaled_progress,
                                   ' ' * (self.size - scaled_progress)))
    sys.stdout.flush()

