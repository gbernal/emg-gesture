import numpy as np

fps = 29.97
offset = 65 
windowError = 20 # frames

def get_training_windows(fname):
  data = np.genfromtxt('annotation/' + fname + '.csv', delimiter=',', skip_header=1,names=['start', 'end'])
  start = map(lambda x: (x - offset - windowError) * 1000 / fps, data['start'])
  end = map(lambda x: (x - offset  + windowError) * 1000 / fps, data['end'])
  windows = zip(start, end)
  return windows
