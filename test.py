import torch
from torch.utils.tensorboard import SummaryWriter
from config import Config
from torchvision.transforms import transforms

detransform = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[2, 2, 2]),
    transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1])
])
writer = SummaryWriter(log_dir='./log')
opt = Config()
noises = torch.randn(opt.batch_size, opt.nz, 1, 1)
noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
noises = noises.to('cuda:0')
NetG = torch.load('./Module_G/NetG-490.pth')
img = NetG(noises)
img = img.to('cpu')
detransform(img)
writer.add_images('test1', img, 0)
writer.close()
