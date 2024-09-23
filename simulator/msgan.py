from torch import nn


class MSG_Generator(nn.Module):
    def __init__(self, img_channels=3, input_h_dim=256, start_h_dim=512, layers_per_block=5):
        super(MSG_Generator, self).__init__()
        self.start_h_dim = start_h_dim

        #self.init_size = (25, 34)  # Initial size before upsampling

        self.pre_conv = nn.Conv2d(in_channels=input_h_dim, out_channels=start_h_dim, kernel_size=3, stride=1, padding=1)

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(start_h_dim),
            nn.Upsample(scale_factor=2)
        )

        for i in range(layers_per_block - 1):
            self.block1.add_module(f'conv_{i+1}', nn.Conv2d(start_h_dim, start_h_dim, kernel_size=3, stride=1, padding=1))
            self.block1.add_module(f'bn_{i+1}',nn.BatchNorm2d(start_h_dim, 0.8))
            self.block1.add_module(f'lrelu_{i+1}',nn.LeakyReLU(0.2, inplace=True))
        self.block1.add_module(f'conv_{layers_per_block}', nn.Conv2d(start_h_dim, start_h_dim, kernel_size=3, stride=1, padding=1))
        self.block1.add_module(f'bn_{layers_per_block}', nn.BatchNorm2d(start_h_dim, 0.8))
        self.block1.add_module(f'lrelu_{layers_per_block}', nn.LeakyReLU(0.2, inplace=True))

        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2)
        )
        for i in range(layers_per_block - 1):
            self.block2.add_module(f'conv_{i+1}', nn.Conv2d(start_h_dim, start_h_dim, kernel_size=3, stride=1, padding=1))
            self.block2.add_module(f'bn_{i+1}',nn.BatchNorm2d(start_h_dim, 0.8))
            self.block2.add_module(f'lrelu_{i+1}',nn.LeakyReLU(0.2, inplace=True))
        self.block2.add_module(f'conv_{layers_per_block}', nn.Conv2d(start_h_dim, start_h_dim // 2, kernel_size=3, stride=1, padding=1))
        self.block2.add_module(f'bn_{layers_per_block}', nn.BatchNorm2d(start_h_dim // 2, 0.8))
        self.block2.add_module(f'lrelu_{layers_per_block}', nn.LeakyReLU(0.2, inplace=True))

        self.block3 = nn.Sequential(
            nn.Upsample(scale_factor=2)
        )
        for i in range(layers_per_block - 1):
            self.block3.add_module(f'conv_{i+1}', nn.Conv2d(start_h_dim // 2, start_h_dim // 2, kernel_size=3, stride=1, padding=1))
            self.block3.add_module(f'bn_{i+1}',nn.BatchNorm2d(start_h_dim // 2, 0.8))
            self.block3.add_module(f'lrelu_{i+1}',nn.LeakyReLU(0.2, inplace=True))
        self.block3.add_module(f'conv_{layers_per_block}', nn.Conv2d(start_h_dim // 2, start_h_dim // 4, kernel_size=3, stride=1, padding=1))
        self.block3.add_module(f'bn_{layers_per_block}', nn.BatchNorm2d(start_h_dim // 4, 0.8))
        self.block3.add_module(f'lrelu_{layers_per_block}', nn.LeakyReLU(0.2, inplace=True))

        self.final_conv1 = nn.Conv2d(start_h_dim, img_channels, kernel_size=3, stride=1, padding=1)
        self.final_conv2 = nn.Conv2d(start_h_dim // 2, img_channels, kernel_size=3, stride=1, padding=1)
        self.final_conv3 = nn.Conv2d(start_h_dim // 4, img_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        #out = self.l1(z)
        #out = out.view(out.shape[0], self.start_h_dim, self.init_size[0], self.init_size[1])

        out = self.pre_conv(z)

        out1 = self.block1(out)
        img1 = self.tanh(self.final_conv1(out1))

        out2 = self.block2(out1)
        img2 = self.tanh(self.final_conv2(out2))

        out3 = self.block3(out2)
        img3 = self.tanh(self.final_conv3(out3))

        return [img1, img2, img3]