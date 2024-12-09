import torch
import torch.nn as nn

# one down-sampling conv block
def conv_block(in_channels, out_channels, use_instancenorm=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
    if use_instancenorm:
        layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)

# one up-sampling conv block
def deconv_block(in_channels, out_channels, use_dropout=False):
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
              nn.InstanceNorm2d(out_channels),
              nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

# global self-attention
class self_attention(nn.Module):
    def __init__(self, in_channels):
        super(self_attention, self).__init__()
        # key, query, value
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # scaling param
        self.gamma = nn.Parameter(torch.zeros(1))
        # softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()

        # compute query
        queries = self.query(x).view(B, -1, H*W) # [B, C, H, W] --> [B, C//8, H*W]
        queries = queries.permute(0, 2, 1) # [B, C//8, H*W] --> [B, H*W, C//8]

        # compute keys
        keys = self.key(x).view(B, -1, H*W) # [B, C, H, W] --> [B, C//8, H*W]

        # compute attention
        energy = torch.bmm(queries, keys) # [B, H*W, C//8] x [B, C//8, H*W] --> [B, H*W, H*W]
        attention = self.softmax(energy / (C//8)**0.5) # attention with scale factor sqrt(dim)

        # compute values
        values = self.value(x).view(B, -1, H*W) # [B, C, H, W] --> [B, C, H*W]

        # apply attention
        out = torch.bmm(attention, values.permute(0, 2, 1)) # [B, H*W, H*W] x [B, H*W, C] --> [B, H*W, C]
        out = out.permute(0, 2, 1) # [B, H*W, C] --> [B, C, H*W]
        out = out.view(B, C, H, W) # reshape

        # rescale, residule, return
        return self.gamma * out + x

# model
class Generator(nn.Module):
    """
    U-Net Generator. Encoder-decoder structure. Down sampling followed by upsampling.
    """
    def __init__(self, input_channels=3, output_channels=3):
        super(Generator, self).__init__()

        # downsampling path
        self.enc1 = conv_block(input_channels, 64, use_instancenorm=False) # (3, 256, 256) -> (64,)
        self.enc2 = conv_block(64, 128) # (64, 128, 128) -> (128,)
        self.enc3 = conv_block(128, 256) # (128, 64, 64) -> (256,)
        self.enc4 = conv_block(256, 512) # (256, 32, 32) -> (512,)
        self.enc5 = conv_block(512, 512) # (512, 16, 16) -> (512,)
        self.enc6 = conv_block(512, 512) # (512, 8, 8) -> (512,)
        self.enc7 = conv_block(512, 512) # (512, 4, 4) -> (512,)
        self.enc8 = conv_block(512, 512, use_instancenorm=False) # (512, 2, 2) -> (512, 1, 1)
        
        # self attention
        self.att1 = self_attention(512) # for enc4
        self.att2 = self_attention(512) # for enc5

        # upsampling path
        self.dec1 = deconv_block(512, 512, use_dropout=True) # no skip connection
        self.dec2 = deconv_block(1024, 512, use_dropout=True) # 512 channel skip with e7
        self.dec3 = deconv_block(1024, 512, use_dropout=True) # 512 channel skip with e6
        self.dec4 = deconv_block(1024, 512) # 512 channel skip with e5
        self.dec5 = deconv_block(1024, 256) # 512 channel skip with e4
        self.dec6 = deconv_block(512, 128) # 256 channel skip with e3
        self.dec7 = deconv_block(256, 64) # 128 channel skip with e2
        self.dec8 = deconv_block(128, 64) # 64 channel skip with e1
        
        # final output layer
        self.final = nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    # forward pass
    def forward(self, x):
        # encoder + self-attention
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        e4 = self.enc4(e3)
        e4 = self.att1(e4)

        e5 = self.enc5(e4)
        e5 = self.att2(e5)

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
        d8 = self.dec8(d7)

        # final layer
        output = self.final(d8)
        output = self.tanh(output)

        # return output
        return output

# discriminator
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        # discriminator receives input and output images concatenated together (channel_size=6)
        self.model = nn.Sequential(
            # input (6, w, h)
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            # no activation function since we will use nn.BCEwithlogitsloss()
        )

    def forward(self, x):
        # forward pass
        return self.model(x)


class _ResidualBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(_ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels_out)
        )

        if channels_in != channels_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1),
                nn.InstanceNorm2d(channels_out)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        if hasattr(self, 'shortcut'):
            residual = self.shortcut(x)
        out += residual
        return out

class Predictor(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=4):
        super(Predictor, self).__init__()

        # Initial downsampling
        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Second downsampling layer
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Residual blocks
        list_residual_blocks = [_ResidualBlock(256, 256) for _ in range(num_residual_blocks)]
        self.residual_blocks = nn.Sequential(*list_residual_blocks)

        # Final upsampling
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.downsampling(x)
        out = self.residual_blocks(out)
        out = self.upsampling(out)
        return out