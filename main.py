import torch
import torchvision as tv
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import Config
from generator import NetG
from discriminator import NetD
from torchvision.transforms import transforms

# define config
opt = Config()

# define tensorboard save path
writer = SummaryWriter(log_dir='./log')
j = 1

# define how transform
transform = tv.transforms.Compose([
    tv.transforms.Resize(opt.img_size),
    tv.transforms.CenterCrop(opt.img_size),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

detransform = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[2, 2, 2]),
    transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1])
])

# define data shuffle
dataset = tv.datasets.ImageFolder(opt.data_path, transform)

# define dataloader
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)

# define/load Net

if opt.load_net_bool:
    NetG = torch.load(opt.Netg_path)
    NetD = torch.load(opt.Netd_path)
    # NetG.load_state_dict(torch.load(opt.Netg_path))
    # NetD.load_state_dict(torch.load(opt.Netd_path))
else:
    NetG = NetG(opt)
    NetD = NetD(opt)

# define optimizer and loss
optimizer_g = torch.optim.Adam(NetG.parameters(), opt.lr)
optimizer_d = torch.optim.Adam(NetD.parameters(), opt.lr)
criterion = torch.nn.BCELoss()

true_label = torch.ones(opt.batch_size)
fake_label = torch.zeros(opt.batch_size)
fix_noises = torch.randn(opt.batch_size, opt.nz, 1, 1)
noises = torch.randn(opt.batch_size, opt.nz, 1, 1)

# chose gpu
if opt.use_gpu:
    NetG = NetG.cuda(0)
    NetD = NetD.cuda(0)
    criterion.cuda(0)
    true_label = true_label.to('cuda:0')
    fake_label = fake_label.to('cuda:0')
    fix_noises = fix_noises.to('cuda:0')
    noises = noises.to('cuda:0')


for i in range(0,1000):
    for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
        if opt.use_gpu:
            img = img.to('cuda:0')
        if (ii + 1) % opt.d_every == 0:
            optimizer_d.zero_grad()
            real_img = img
            output = NetD(real_img)
            error_d_real = criterion(output, true_label)
            error_d_real.backward()

            noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
            fake_img = NetG(noises)
            fake_output = NetD(fake_img)
            error_d_fake = criterion(fake_output, fake_label)
            error_d_fake.backward()
            optimizer_d.step()

        if (ii + 1) % opt.g_every == 0:
            optimizer_g.zero_grad()
            noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
            fake_img = NetG(noises)
            fake_output = NetD(fake_img)

            error_g = criterion(fake_output, true_label)
            error_g.backward()
            optimizer_g.step()

    if i % 50 == 0:
        print(f'epoch:{i+2030}')
        noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
        img = NetG(noises)
        img = img.to('cpu')
        detransform(img)
        writer.add_images('train', img, j)
        j += 1
        torch.save(NetG, f'./Module_G/NetG-{int(i+2030)}.pth')
        torch.save(NetD, f'./Module_D/NetD-{int(i+2030)}.pth')