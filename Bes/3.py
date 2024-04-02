#%%
import utils
import model
import dataset
import torch



torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)
torch.manual_seed(42)
ds = dataset.Dataset()



tr_ds, ts_ds = torch.utils.data.random_split(ds, [1000, 9000])
tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=32, shuffle=True)
ts_dl = torch.utils.data.DataLoader(ts_ds, batch_size=32, shuffle=False)



yolo = model.YOLO(dm=4)
print(f"Number of parameters: {sum(p.numel() for p in yolo.parameters())}")
optimizer = torch.optim.Adam(yolo.parameters(), lr=0.001)
num_epochs = 300



def criterion(out, lbl):
  c_error = torch.nn.functional.binary_cross_entropy(out[:, 0], lbl[:, 0])
  x_error = (((lbl[:, 1] - out[:, 1])[lbl[:, 0] == 1]) ** 2).sum()
  y_error = (((lbl[:, 2] - out[:, 2])[lbl[:, 0] == 1]) ** 2).sum()
  r_error = (((lbl[:, 3] - out[:, 3])[lbl[:, 0] == 1]) ** 2).sum()
  return c_error + x_error + y_error + r_error



for epoch in range(num_epochs):
  for img, lbl in tr_dl:
    optimizer.zero_grad()
    out = yolo(img)
    loss = criterion(out, lbl)
    loss.backward()
    optimizer.step()
  print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")



img = tr_ds[0][0]
out = yolo(img)
out = out.squeeze().clone().detach().numpy().round(2)
utils.array_to_plot(out, img.squeeze())



img = tr_ds[1][0]
out = yolo(img)
out = out.squeeze().clone().detach().numpy().round(2)
utils.array_to_plot(out, img.squeeze())
# %%
img = ts_ds[8007][0]
out = yolo(img)
out = out.squeeze().clone().detach().numpy().round(2)
utils.array_to_plot(out, img.squeeze())