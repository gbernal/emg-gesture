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
    data = center_data(data)

  gestures = []
  for fname in gesture_annotation_files:
    gesture = create_gesture_from_filename(fname, data)
    gestures.append(gesture)
  return gestures

def center_data(data):
  data[time_field] = data[time_field] - data[time_field][0]
  for ax in axis:
    floating_center = 464 
    # TODO: do this on the fly
    # determined by taking average of all samples
    avg = np.average(data[ax])
    data[ax] -= floating_center
    data[ax] = np.abs(data[ax])
  return data

def get_training_data_windows(fname):
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
  return windows

def create_gesture_from_filename(fname, data):
  windows = get_training_data_windows(fname)
  window_size = windows[0][1] - windows[0][0]
  gesture = Gesture(fname)
  # set data so that it starts at 0
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
  import time
  detector = GestureDetector(gestures)
  for fname in files:
    data = np.genfromtxt('data/' + fname + '.CSV', delimiter=',', skip_header=1, 
        names=[time_field] + axis)
    data = center_data(data)
    t0 = time.clock()
    detector.find_gestures(data)
    print 'took', time.clock() - t0, 's'
 
def test_training_gestures(gestures):
  detector = GestureDetector(gestures)
  windows = []
  for ann_fname in gesture_annotation_files:
    windows += get_training_data_windows(ann_fname)
  for fname in files:
    data = np.genfromtxt('data/' + fname + '.CSV', delimiter=',', skip_header=1, 
        names=[time_field] + axis)
    data = center_data(data)
    for start, end in windows:
      found_gesture = False
      best_score = 0
      best_gesture = None
      for gesture in gestures:
        is_gesture, score = detector.is_gesture(data, start, gesture)
        if score > best_score:
          if is_gesture:
            print 'found gesture', gesture, 'in training data at time', start, 'with score', score
            found_gesture = is_gesture
          else:
            best_score = score
            best_gesture = gesture
      if not found_gesture:
        print 'did not find a gesture at time', start, '. But got a best score of', best_score, 'for', best_gesture
