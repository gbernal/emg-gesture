import numpy as np
from numpy.lib.recfunctions import append_fields
import sys

'''
Represents a single gesture as given by signals
TODO: Come up with a more time efficient manner of storing a gesture for later detection
'''
class Gesture:
  _gestureNumber = 1
  def __init__(self, name=None): 
    if name:
      self.name = name
    else:
      self.name = "Gesture " + Gesture._gestureNumber

    print "Initializing", self.name

    self.gestureNumber = Gesture._gestureNumber
    Gesture._gestureNumber += 1

    # TODO: There has to be a better way to handle these
    self._time_field = 'millis'
    self._range_field = 'range'
    self._sensor_fields = ['sens0', 'sens1', 'sens2']

    self._training_data = {}
    self._interlaced_training_data = {}
    self._interlaced_data_scores = {}
    # The amount of time the training data should be normalized to (in ms)
    self._gesture_time = 100.0 
    self._max_training_gesture_time = 0.0

  '''
  Returns whether this Gesture model has been given training data
  '''
  def has_been_trained(self):
    return len(self._training_data) > 0

  '''
  Train the gesture recognition model with specific training data. There is a preprocessing
  stage where the data is manipulated to be stored as training data in a good form. The data
  is stored, and then there is an additional post processing stage where the data is augmented to
  provide additional information for future matching algorithms
  '''
  def train(self, data, time_field=None, sensor_fields=None):
    if time_field is None: time_field = self._time_field
    if sensor_fields is None: sensor_fields = self._sensor_fields
    # TODO: Rename sensor and time fields if they are different in data than the member variables
    for sensor_field in data:
      sensor_data = self._preprocess(data[sensor_field], time_field, sensor_field)
      if sensor_field in self._training_data:
        self._training_data[sensor_field].append(sensor_data)
      else:
        self._training_data[sensor_field] = [sensor_data]

    for sensor_field in self._sensor_fields:
      interlaced_training_data = np.sort(np.concatenate(tuple(
          self._training_data[sensor_field])))
      interlaced_training_data[sensor_field] = self._movingaverage(
          interlaced_training_data[sensor_field], min(50, int(len(interlaced_training_data)/10.)))
      self._interlaced_training_data[sensor_field] = interlaced_training_data

    for sensor_field in data:
      sensor_data = self._postprocess(data[sensor_field], time_field, sensor_field)

    return self._training_data


  '''
  A step to normalize the time domain to range from 0 to self._gesture_time.

  Currently the length of the gesture is arbitrarily set. In the future, it might be beneficial to
  set the time length of the gesture to the average length of the training data, though probably not
  since all incoming data will be also stretched to conform around the average length of the 
  training data
  '''
  def _normalize_time_domain(self, data, time_field):
    low = 0.0
    mins = min(data[time_field])
    maxs = max(data[time_field])
    rng = maxs - mins
    data[time_field] = self._gesture_time - (((self._gesture_time - low) * (maxs -
        data[time_field])) / rng)
    return data

  '''
  The preprocessing state for data that is added to the gesture for training
  '''
  def _preprocess(self, data, time_field, sensor_field):
    # center data at 0 and amplify data based on largest wave form
    data = np.copy(data)
    data = self._normalize_time_domain(data, time_field)

    data[sensor_field] = self._movingaverage(data[sensor_field], 10)
    '''
    Filters not currently working on this data
    blp, alp = signal.butter(4, 10/40, 'low', analog=True) # add 10Hz LP
    bhp, ahp = signal.butter(4, .0, 'high', analog=True) # add 80Hz HP
    bn, an = signal.butter(6, [(58/40)/(FS/2), (62/40)/(FS/2)], 'bandstop', analog=True) # add 80Hz HP
    data[sensor] = signal.lfilter(bhp, ahp, data[sensor])
    data[sensor] = signal.lfilter(bn, an, data[sensor])
    data[sensor] = signal.lfilter(blp, alp, data[sensor])
    '''
    data = self._cleanse(data, time_field, sensor_field)
    data = self._add_range_field_to_training_data(data, sensor_field)
    return data 

  def _postprocess(self, data, time_field, sensor_field):
    self._add_range_to_training_data(sensor_field)

    # Update the max training gesture time
    self._max_training_gesture_time = max(self._max_training_gesture_time,
        data[time_field][-1] - data[time_field][0])

    # set the standards for an ideal gesture by using a percentage of the score of the average of
    # the training data
    self._interlaced_data_scores[sensor_field] = self._compare_sensor(self._interlaced_training_data[sensor_field], sensor_field)
    print self._interlaced_data_scores

  def _cleanse(self, data, time_field, sensor_field):
    return data[[time_field, sensor_field]]

  def _add_range_field_to_training_data(self, data, sensor_field): 
    # augment data with range information
    data = append_fields(data, self._range_field, data=np.zeros(data[self._time_field].shape, 
        dtype=data[sensor_field].dtype))
    return data

  '''
  Adds actual range information to the training data by looking at the interlaced data for
  each of the sensors
  '''
  def _add_range_to_training_data(self, sensor_field): 
    all_samples = self._interlaced_training_data[sensor_field]
    delta = .1
    ranges = {}
    for sample in all_samples:
      ranges[sample[self._time_field]] = max(all_samples[(sample[
          self._time_field] + delta > all_samples[self._time_field]) & 
          (sample[self._time_field] - delta < all_samples[self._time_field])][sensor_field])
    for data in self._training_data[sensor_field]:
      for i in data:
        if i[self._time_field] in ranges:
          i[self._range_field] = ranges[i[self._time_field]]
    return self._training_data

 
  '''
  Provides a moving average function to smooth out signals
  '''
  def _movingaverage(self, data, window_len=10):
    if data.ndim != 1:
      raise ValueError, "movingaverage only accepts 1 dimension arrays."
    if data.size < window_len:
      raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
      return data 
    s = np.r_[2*data[0]-data[window_len:1:-1], data, 2*data[-1]-data[-1:-window_len:-1]]
    w = np.ones(window_len, 'd')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]

  '''
  Compares some data to the trained gesture and returns some number between 0 and 1
  representing how ideal of a signal it is based on the training data, where 0 is least ideal
  and 1 is most ideal.
  '''
  def compare(self, signal):
    # TODO: Use various stretching/panning methods
    signal = self._normalize_time_domain(signal, self._time_field)
    dist = 0
    for sensor_field in self._sensor_fields:
      dist += self._compare_sensor(signal, sensor_field)
    return dist / float(sum(self._interlaced_data_scores.values()))

  '''
  Compares a signal over a specific sensor
  '''
  def _compare_sensor(self, signal, sensor_field):
    training_data = self._training_data[sensor_field]
    dist = 0
    for trial in training_data:
      i = 0
      j = 0
      while i < len(trial) and j < len(signal):
        training_sample = trial[i]
        signal_sample = signal[j]
        training_range = training_sample[self._range_field]
        high = training_sample[sensor_field] + training_range
        low = training_sample[sensor_field] - training_range
        if signal_sample[sensor_field] >= low and signal_sample[sensor_field] <= high:
          dist += training_range - abs(training_sample[sensor_field] - signal_sample[sensor_field])

        i += 1
        # While the next index of the signal is closer to the next index of the data its being 
        # compared to, increment that index of the signal
        while i < len(trial) and j + 1 < len(signal) and abs(signal[j + 1][self._time_field] - trial[i][self._time_field]) <= abs(signal[j][self._time_field] - trial[i][self._time_field]):
          j += 1
    return dist
  
  ''' GETTERS '''
  def get_time_field(self):
    return self._time_field

  def get_range_field(self):
    return self._range_field

  def get_sensor_fields(self):
    return self._sensor_fields

  def get_gesture_time(self):
    return self._gesture_time

  def get_max_gesture_time_length(self):
    # TODO: increase this multiplier, probably
    return self._max_training_gesture_time * 1

  def get_data(self):
    return self._training_data

  def get_interlaced_data(self):
    return self._interlaced_training_data

  def __repr__(self):
    return self.name 

  def __str__(self):
    return self.name 
