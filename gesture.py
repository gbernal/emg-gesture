import numpy as np
from numpy.lib.recfunctions import append_fields
import sys
'''
Represents a single gesture as given by signals
'''
class Gesture:
  _gestureNumber = 1
  def __init__(self, name=None): 
    if name:
      self.name = name
    else:
      self.name = "Gesture " + Gesture._gestureNumber

    self.gestureNumber = Gesture._gestureNumber
    Gesture._gestureNumber += 1

    self._time_field = 'millis'
    self._range_field = 'range'
    self._sensor_fields = ['sens0', 'sens1', 'sens2']

    self._training_data = {}
    self._interlaced_training_data = {}
    # The amount of time the training data should be normalized to (in seconds)
    self.gesture_time = 100.0 

  def has_been_trained(self):
    return len(self._training_data) > 0

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


  def _normalize_time_domain(self, data, time_field):
    data[time_field] = data[time_field] - data[time_field][0]
    low = 0.0
    mins = min(data[time_field])
    maxs = max(data[time_field])
    rng = maxs - mins
    data[time_field] = self.gesture_time - (((self.gesture_time - low) * (maxs -
        data[time_field])) / rng)
    return data

  def _preprocess(self, data, time_field, sensor_field):
    # center data at 0 and amplify data based on largest wave form
    data = self._normalize_time_domain(data, time_field)
    maxes = []
    avg = np.average(data[sensor_field])
    # TODO: do this on the fly
    # determined by taking average of all samples
    floating_center = 464
    data[sensor_field] -= floating_center
    data[sensor_field] = np.abs(data[sensor_field])

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

  def _cleanse(self, data, time_field, sensor_field):
    return data[[time_field, sensor_field]]

  def _add_range_field_to_training_data(self, data, sensor_field): 
    # augment data with range information
    data = append_fields(data, self._range_field, data=np.zeros(data[self._time_field].shape, 
        dtype=data[sensor_field].dtype))
    return data

  def _add_range_to_training_data(self, sensor_field): 
    all_samples = self._interlaced_training_data[sensor_field]
    delta = .1
    ranges = {}
    for sample in all_samples:
      ranges[sample[self._time_field]] = max(all_samples[(sample[self._time_field] + delta >
          all_samples[self._time_field]) & (sample[self._time_field] - delta <
              all_samples[self._time_field])][sensor_field])
    for data in self._training_data[sensor_field]:
      for i in data:
        if i[self._time_field] in ranges:
          i[self._range_field] = ranges[i[self._time_field]]
    return self._training_data

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

  def get_time_field(self):
    return self._time_field

  def get_range_field(self):
    return self._range_field

  def get_sensor_fields(self):
    return self._sensor_fields

  def get_data(self):
    return self._training_data

  def get_interlaced_data(self):
    return self._interlaced_training_data
