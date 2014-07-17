class GestureDetector:
  def __init__(self, gestures=[]):
    self._gestures = gestures

  def add_gesture(self, gesture):
    self._gestures.append(gesture)

  def find_gestures(self, data):
    for gesture in self._gestures:
      for row in data[::10]:
        # This could potentially be very slow. Might want to clean this up with code, especially
        # given than millis is always increasing
        compare = data[(data[gesture._time_field] >= row[gesture._time_field]) &
            (data[gesture._time_field] <= row[gesture._time_field] + 
                gesture.get_max_gesture_time_length()) & (row[gesture._time_field] + 
                gesture.get_max_gesture_time_length() <= data[gesture._time_field][-1])]
        if len(compare) > 0:
          score = gesture.compare(compare)
          # not working currently
          if score > 1:
            print 'found', gesture, 'at', row[gesture._time_field]

  '''GETTERS'''

  def get_gestures(self):
    return self._gestures
