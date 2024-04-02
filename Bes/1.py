import torch
import imageio as iio
import pandas as pd
import utils as u



class Dataset(torch.utils.data.Dataset):
  def __init__(self):
    self.labels = pd.read_parquet('./128x128_10k/labels.parquet')
    self.images = pd.read_parquet('./128x128_10k/images.parquet')

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img = self.images.iloc[idx]['image']
    img = iio.v3.imread(img)
    img = torch.tensor(img).float()
    img = torch.matmul(img[...,:3], torch.tensor([0.2989, 0.5870, 0.1140]))
    lbl = self.labels[(self.labels['image_id'] == idx)]
    lbl = lbl[['x', 'y', 'radius', 'class']]
    lbl = self.label_to_grid(lbl)
    img = img.unsqueeze(0)
    return img, lbl

  def label_to_grid(self, coords, grid_size=8):
    res = torch.zeros(4, grid_size, grid_size).float()
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
    return res



if __name__ == '__main__':
  ds = Dataset()
  img, lbl = ds[88]
  print(lbl, img.shape, lbl.shape)
  lbl = lbl.clone().detach().numpy()
  u.array_to_plot(lbl, img.squeeze())