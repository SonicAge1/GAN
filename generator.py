from torch import nn


# define GanNet
class NetG(nn.Module):
    def __init__(self, opt):
        super(NetG, self).__init__()
        self.ngf = opt.ngf
        self.main = nn.Sequential(
            nn.ConvTranspose2d(opt.nz, self.ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),  # 4*4*512
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*8, self.ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 8*8*256
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 16*16*128
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, kernel_size=4, stride=2, padding=1, bias=False),  # 32*32*64
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),  # 64*64*3
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

