import numpy as np
from gesture import *
from plot_gesture import *

from gesture_detector import GestureDetector

gesture_annotation_files = ['hand-close', 'hand-open']

files = ['Guillermo2ndRec']

time_field = 'millis'
axis = ['sens0', 'sens1', 'sens2']

fps = 29.97
offset = 65 
windowError = 20 # frames

timeshift0 = 0
timeshift1 = 506
timeshift2 = 642
sensors_and_shift = zip(axis, [timeshift0, timeshift1, timeshift2])
 
def create_gestures():
  for fname in files:
    data = np.genfromtxt('data/' + fname + '.CSV', delimiter=',', skip_header=1, 
        names=[time_field] + axis)

  gestures = []
  for fname in gesture_annotation_files:
    gestures.append(create_gesture_from_filename(fname, data))
  return gestures

def create_gesture_from_filename(fname, data):
  # Load the video annotation file
  annotation = np.genfromtxt('annotation/' + fname + '.csv', delimiter=',', skip_header=1,
      names=['start', 'end'])
  # Get the windows in the time domain for which the annotation file describes the gesture in the
  # data
  # windows are length 2 tuples where the first number represents the beginning of a window and the
  # second number represents the end of th a window, in the time domain
  start = map(lambda x: (x - offset - windowError) * 1000 / fps, annotation['start'])
  end = map(lambda x: (x - offset  + windowError) * 1000 / fps, annotation['end'])
  windows = zip(start, end)
  window_size = windows[0][1] - windows[0][0]
  gesture = Gesture(fname)
  # set data so that it starts at 0
  data[time_field] = data[time_field] - data[time_field][0]
  for window in windows:
    filtered_trial_data = {}
    for sensor,shift in sensors_and_shift:
      filtered_trial_data[sensor] = np.copy(data[(data[time_field] + shift > window[0]) & 
          (data[time_field] + shift < window[1])])
    gesture.train(filtered_trial_data)
  return gesture

def plot_gestures(gestures):
  for gesture in gestures:
    plot_gesture(gesture)

def detect_gestures(gestures):
  detector = GestureDetector(gestures)
  for fname in files:
    data = np.genfromtxt('data/' + fname + '.CSV', delimiter=',', skip_header=1, 
        names=[time_field] + axis)
    detector.find_gestures(data)
 
