# Hand Gesture Recognition
Gesture detection based on EMG data recordings.

## Usage
Visualize a semi-accurate heat map of the training data:
```
python visualize.py
```

Run gesture detection on the training data to ensure that those spots in the data are recognized as the correct gestures:
```
python test_training.py
```

Run gesture detection on all of the data file (from which the training data is extracted) to find the instances that the specific gestures are determined
```
python detect.py
```

## Known Issues
1. The speed of the gesture detection algorithm is not fast enough. Currently it is evaluating 1/10 samples (once every .4s) and it's still about a factor of 2 too slow.
2. Only running gesture detection once every .4s.
3. The training data/annotations are stretched to create gesture profiles that are somewhat time independent (gestures can take a variable amount of time). However, gestures are only evaluated in their real time frame

