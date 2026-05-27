import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# 
# Encoder block
# - Uses 2x2 max pooling with stride 2 to reduce spatial resolution
# - Two outputs, pre-max pool output value used for skip connections
#
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(DownBlock, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.dropout_1 = nn.Dropout2d(p=dropout_rate)
        self.dropout_2 = nn.Dropout2d(p=dropout_rate)
    def forward(self, x):
        x = self.dropout_2(self.conv_2(self.dropout_1(self.conv_1(x))))
        return self.max_pool(x), x


#
# Decoder block
# - uses nearest neighbor upsampling to increase spatial resolution
# - takes a skip connection input for forward evaluation
#
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(UpBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConstantPad2d((0,1,0,1), 0),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.dropout_1 = nn.Dropout2d(p=dropout_rate)
        self.dropout_2 = nn.Dropout2d(p=dropout_rate)
    def forward(self, x, x_skip):
        x = torch.cat((self.upsample(x), x_skip), 1)
        return self.dropout_2(self.conv_2(self.dropout_1(self.conv_1(x))))
    
#
# "Bottom" block
# - used for the bottom of the U-shape
# - does not affect spatial resolution
# - does not produce or take skip connections
#
class BottomBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(BottomBlock, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.dropout_1 = nn.Dropout2d(p=dropout_rate)
        self.dropout_2 = nn.Dropout2d(p=dropout_rate)
    def forward(self, x):
        return self.dropout_2(self.conv_2(self.dropout_1(self.conv_1(x))))
#
# Model definition
# - 3 encoders, the bottom block, 3 decoders
# - 5 input channels (RGB + DSM + DEM), 2 output channels (negative & positive label logits)
# - hidden layer channel count doubled in each encoder block (incl. bottom), halved in each decoder
# - crop applied before final output layer, removing `crop` pixels from each edge
#
class UNet(nn.Module):
    def __init__(self, crop=8, f=32):
        super(UNet, self).__init__()
        self.down_0 = DownBlock(5, f*1)
        self.down_1 = DownBlock(f*1, f*2)
        self.down_2 = DownBlock(f*2, f*4)
        self.bottom = BottomBlock(f*4, f*8)
        self.up_0 = UpBlock(f*8, f*4)
        self.up_1 = UpBlock(f*4, f*2)
        self.up_2 = UpBlock(f*2, f*1)
        
        # final output layer (f channels -> 2 channels)
        self.logits = nn.Sequential(
            nn.Conv2d(in_channels=f*1, out_channels=2,
                      kernel_size=1),
        )
        self.crop = crop

        # correctly formed softmax function defined here for convenience, not used in the model forward pass
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # skip connections saved for use in encoder blocks
        x, skip_0 = self.down_0(x)
        x, skip_1 = self.down_1(x)
        x, skip_2 = self.down_2(x)
        
        x = self.bottom(x)
        
        # skip connections applied in reverse order
        x = self.up_0(x, skip_2)
        x = self.up_1(x, skip_1)
        x = self.up_2(x, skip_0)
        
        crop = self.crop
        
        # last two axes are cropped from both ends 
        logits = self.logits(x[:,:,crop:-crop,crop:-crop])
        
        # returning logits, not probabilities (softmax needs to be applied to produce probabilities)
        return logits
