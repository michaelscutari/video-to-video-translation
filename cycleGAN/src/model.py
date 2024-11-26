import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.local_route = nn.Sequential(
            # padding largens image; conv reduces it back to original size.
            nn.ReflectionPad2d(1), # adds one-pixel border around image (reflected outwards)
            nn.Conv2d(in_features, in_features, 3), #in_channels=out_channels bc it's the same image channels in & out.  3x3 Kernel.
            nn.InstanceNorm2d(in_features), #normalizes the input per channel
            
            nn.ReLU(inplace=True), #"inplace" :. will modify the input directly, w/o allocating additional output. memory efficient.
            
            nn.ReflectionPad2d(1), 
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)

            # no ReLU here.  ReLU is applied after sum of local_route & highway
        )
    
    def forward(self, x):
        return x + self.local_route(x) # x bypasses local_route (residual) to help backpropagation.
    
class Generator(nn.Module):
    def __init__(self, input_num_channels=3, output_num_channels=3, n_residual_blocks=9):
        super(Generator, self).__init__()

        ## Initial Convolution Block
        model = [ 
            nn.ReflectionPad2d(3), #adds 3-pixel border (reflected outwards)
            nn.Conv2d(input_num_channels, 64, 7), #extracts 64 diff features using 7x7 kernel
            nn.InstanceNorm2d(64), #normalize data along each channel
            nn.ReLU(inplace=True)
        ]

        ## Downsample
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):              # 2 downsampling layers
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), #doubles features; halves image size
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        ## Residual Blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(out_features)] # processing in deep feature space

        ## Upsample
        out_features = in_features // 2
        for _ in range(2):              # 2 upsampling layers
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), #halves features; doubles image size
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        ## Output Layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_num_channels, 7),
            nn.Tanh() #squashes output to [-1, 1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_num_channels=3):
        super(Discriminator, self).__init__()

        ## Multiple convolutions
        model = [
            nn.Conv2d(input_num_channels, 64, 4, stride=2, padding=1), # 64 features, 4x4 kernel, ~halves image size
            nn.LeakyReLU(.2, inplace=True), #slope of negative part is y=.2x
        ]
        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # double features, half image size
            nn.InstanceNorm2d(128), # norms per channel
            nn.LeakyReLU(.2, inplace=True)
        ]
        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1), #double features, half image size
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(.2, inplace=True)
        ]
        model += [
            nn.Conv2d(256, 512, 4, stride=2, padding=1), #double features, half image size
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(.2, inplace=True)
        ]

        ## FullyConvNet classification layer
        model += [
            nn.Conv2d(512, 1, 4, padding=1) # report one feature, well-informed by 512 features
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    