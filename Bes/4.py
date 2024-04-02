
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio as iio



df_labels = pd.read_parquet('./128x128_10k/labels.parquet')
df_images = pd.read_parquet('./128x128_10k/images.parquet')


max_orientation = df_labels['orientation'].max()
min_orientation = df_labels['orientation'].min()
print("Max Orientation:", max_orientation)
print("Min Orientation:", min_orientation)



def label_to_grid(coords, grid_size=8):
  res = np.zeros([5, grid_size, grid_size])
  for coord in coords.iterrows():
    x, y = coord[1]['x'], coord[1]['y']
    box_s = 128 / grid_size
    box_x = int(x // (128 / grid_size))
    box_y = int(y // (128 / grid_size))
    off_x = (x - box_x * box_s)
    off_y = (y - box_y * box_s)
    res[0, box_y, box_x] = 1.0
    res[1, box_y, box_x] = off_x / box_s
    res[2, box_y, box_x] = off_y / box_s
    res[3, box_y, box_x] = coord[1]['radius'] / 16.0
    res[4, box_y, box_x] = coord[1]['orientation'] / 6.29
  return res



def array_to_plot(label, image):
  _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
  ax2.imshow(image, cmap='gray')
  ax3.imshow(image, cmap='gray')
  ax1.imshow(np.zeros((128, 128)), cmap='gray')

  for i in range(label.shape[1]):
    for j in range(label.shape[2]):
      fc = 'yellow' if label[0, i, j] == 1 else 'none'
      one = plt.Rectangle((j*16, i*16), 16, 16, linewidth=1, edgecolor='r', facecolor=fc, alpha=0.3)
      two = plt.Rectangle((j*16, i*16), 16, 16, linewidth=1, edgecolor='r', facecolor=fc, alpha=0.3)
      ax1.add_patch(one)
      ax3.add_patch(two)
      c = 'yellow' if label[0, i, j] == 1 else 'red'
      ax1.text(j*16 + 8, i*16 + 8, label[0, i, j], color=c, ha='center', va='center', fontsize=6)
      ax3.text(j*16 + 8, i*16 + 8, label[0, i, j], color=c, ha='center', va='center', fontsize=6)

  for i in range(label.shape[1]):
    for j in range(label.shape[2]):
      if label[0, i, j] == 1:
        ax1.plot(j*16 + label[1, i, j]*16, i*16 + label[2, i, j]*16, 'bo', markersize=2)
        ax2.plot(j*16 + label[1, i, j]*16, i*16 + label[2, i, j]*16, 'bo', markersize=2)
        ax3.plot(j*16 + label[1, i, j]*16, i*16 + label[2, i, j]*16, 'bo', markersize=2)

  for i in range(label.shape[1]):
    for j in range(label.shape[2]):
      if label[0, i, j] == 1:
        ax1.add_artist(plt.Circle((j*16 + label[1, i, j]*16, i*16 + label[2, i, j]*16), label[3, i, j]*16, color='blue', fill=False))
        ax3.add_artist(plt.Circle((j*16 + label[1, i, j]*16, i*16 + label[2, i, j]*16), label[3, i, j]*16, color='blue', fill=False))

  for i in range(label.shape[1]):
    for j in range(label.shape[2]):
      if label[0, i, j] == 1:
        x = j*16 + label[1, i, j]*16
        y = i*16 + label[2, i, j]*16
        r = label[3, i, j]*16
        o = label[4, i, j]*6.29
        ax1.plot([x, x + r*np.cos(o)], [y, y + r*np.sin(o)], 'b-', linewidth=1)
        ax3.plot([x, x + r*np.cos(o)], [y, y + r*np.sin(o)], 'b-', linewidth=1)

  plt.show()



if __name__ == '__main__':
  idx = 6
  label = df_labels[(df_labels['image_id'] == idx)][['x', 'y', 'radius', 'orientation']]
  print(label)
  image = df_images.iloc[idx]['image']
  image = iio.v3.imread(image)
  image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
  label = label_to_grid(label)
  print("label", label)
  array_to_plot(label, image)
