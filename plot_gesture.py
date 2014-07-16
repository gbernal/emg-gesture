import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_gesture(gesture):
  data = gesture.get_data()
  for sensor in gesture.get_sensor_fields():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(sensor)  
    ax.set_xlabel('time')
    ax.set_ylabel('data')
    for trial in data[sensor]:
      data_range = trial[gesture.get_range_field()]
      y1 = trial[sensor] - data_range
      y2 = trial[sensor] + data_range
      ax.fill_between(trial[gesture.get_time_field()], y1, y2, where=y2>=y1,
          facecolor=(1, 0, 0, 1./len(data)), label=ax, interpolate=True, linewidth=0)
    interlaced_data = gesture.get_interlaced_data()
    ax.plot(interlaced_data[sensor][gesture.get_time_field()], interlaced_data[sensor][sensor],
          color='k', label=sensor, linewidth=5)
  plt.show()
