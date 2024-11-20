import torch
import torch.nn as nn

# one down-sampling conv block
def conv_block(in_channels, out_channels, use_batchnorm=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)

# one up-sampling conv block
def deconv_block(in_channels, out_channels, use_dropout=False):
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

# model
class GeneratorUNet(nn.Module):
    """
    U-Net Generator. Encoder-decoder structure. Down sampling followed by upsampling.
    """
    def __init__(self, input_channels=3, output_channels=3):
        super(GeneratorUNet, self).__init__()

        # downsampling path
        self.enc1 = conv_block(input_channels, 64, use_batchnorm=False) # (3, 256, 256) -> (64,)
        self.enc2 = conv_block(64, 128) # (64, 128, 128) -> (128,)
        self.enc3 = conv_block(128, 256) # (128, 64, 64) -> (256,)
        self.enc4 = conv_block(256, 512) # (256, 32, 32) -> (512,)
        self.enc5 = conv_block(512, 512) # (512, 16, 16) -> (512,)
        self.enc6 = conv_block(512, 512) # (512, 8, 8) -> (512,)
        self.enc7 = conv_block(512, 512) # (512, 4, 4) -> (512,)
        self.enc8 = conv_block(512, 512, use_batchnorm=False) # (512, 2, 2) -> (512,)

        # upsampling path
        self.dec1 = deconv_block(512, 512, use_dropout=True) # no skip connection
        self.dec2 = deconv_block(1024, 512, use_dropout=True) # 512 channel skip connection
        self.dec3 = deconv_block(1024, 512, use_dropout=True) # 512 channel skip
        self.dec4 = deconv_block(1024, 512) # 512 channel skip
        self.dec5 = deconv_block(1024, 256) # 512 channel skip
        self.dec6 = deconv_block(512, 128) # 256 channel skip
        self.dec7 = deconv_block(256, 64) # 128 channel skip
        
        # final output layer
        self.final = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    # forward pass
    def forward(self, x):
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # decoder
        d1 = self.dec1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], dim=1)

        # final layer
        output = self.final(d7)
        output = self.tanh(output)

        # return output
        return output

# discriminator
class DiscriminatorPatchGAN(nn.Module):
    def __init__(self, input_channels=6):
        super(DiscriminatorPatchGAN, self).__init__()
        # discriminator receives input and output images concatenated together (channel_size=6)
        self.model = nn.Sequential(
            # input (6, w, h)
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            # no activation function since we will use nn.BCEwithlogitsloss()
        )

    def forward(self, input_image, generated_image):
        # concat images along channel dim
        x = torch.cat((input_image, generated_image), dim=1)
        return self.model(x)
        
