'''
Class responsible for detecting gestures.
'''
class GestureDetector:
  def __init__(self, gestures=[]):
    self._gestures = gestures

  '''
  Adds a gesture to be recognized
  '''
  def add_gesture(self, gesture):
    self._gestures.append(gesture)

  '''
  searches for all gestures in the data using a threshold for the percentage, which is determined by inspection
  '''
  def find_gestures(self, data):
    gestures = []
    for row in data[::10]:
      for gesture in self._gestures:
        is_gesture, score = self.is_gesture(data, row[gesture._time_field], gesture)
        if is_gesture:
          print 'found', gesture, 'at', row[gesture._time_field], 'with score of', score

  def is_gesture(self, data, start, gesture):
    compare = data[(data[gesture._time_field] >= start) &
        (data[gesture._time_field] <= start + gesture.get_max_gesture_time_length()) & 
        (start + gesture.get_max_gesture_time_length() <= data[gesture._time_field][-1])]
    if len(compare) > 0:
      score = gesture.compare(compare)
    else:
      score = 0
    return score >= .5, score

  '''GETTERS'''

  def get_gestures(self):
    return self._gestures
