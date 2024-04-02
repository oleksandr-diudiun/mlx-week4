import torch
import torch.nn as nn



class YOLO(nn.Module):
  def __init__(self, dm=1):
    super(YOLO, self).__init__()
    self.conv1 = nn.Conv2d( 1, 16, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.conv5 = nn.Conv2d(64, dm, kernel_size=3, stride=1, padding=1)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    x = self.conv1(x)
    x = torch.relu(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = torch.relu(x)
    x = self.maxpool(x)
    x = self.conv3(x)
    x = torch.relu(x)
    x = self.maxpool(x)
    x = self.conv4(x)
    x = torch.relu(x)
    x = self.maxpool(x)
    x = self.conv5(x)
    x = torch.sigmoid(x)
    return x



if __name__ == '__main__':
  yolo = YOLO(dm=4)
  print(f"Number of parameters: {sum(p.numel() for p in yolo.parameters())}")
  random_tensor = torch.randn(1, 1, 128, 128)
  output = yolo(random_tensor)
  print(output.shape)
  print(output.squeeze())