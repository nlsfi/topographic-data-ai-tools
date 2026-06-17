import torch
import torch.nn as nn
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class ConvBlock(nn.Module):
    """
    A block of two convolutional layers with batch normalization and ReLU activation.
    This is a standard building block for U-Net architectures.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    The core Attention Gate. It takes a feature map from the encoder (x) and
    a gating signal from the decoder (g) to produce an attention map, which
    is then applied to the encoder's feature map.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Apply the attention map to the feature map from the encoder
        return x * psi
        
class DDSPP(nn.Module):
    """
    Dense Dilated Spatial Pyramid Pooling (DDSPP) module.

    This module captures multi-scale context using a combination of a global
    pooling branch and a series of densely connected dilated convolutions.
    """
    def __init__(self, in_channels, out_channels, branch_channels=256):
        super(DDSPP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # LeakyReLU for all activations
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        # ---- Global Average Pooling Branch ----
        self.global_pool_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # Keras MaxPooling on whole feature map -> Adaptive Pooling
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=True)
        )

        # ---- Dense Dilated Convolution Branches ----
        # Each convolution block: Conv -> BatchNorm -> LeakyReLU
        # The input channels grow with each concatenation
        
        # Branch 1 (Dilation = 2)
        self.conv1 = nn.Conv2d(in_channels + branch_channels, branch_channels, kernel_size=3, 
                               dilation=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(branch_channels)
        
        # Branch 2 (Dilation = 3)
        self.conv2 = nn.Conv2d(in_channels + 2 * branch_channels, branch_channels, kernel_size=3,
                               dilation=3, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(branch_channels)
        
        # Branch 3 (Dilation = 5)
        self.conv3 = nn.Conv2d(in_channels + 3 * branch_channels, branch_channels, kernel_size=3,
                               dilation=5, padding=5, bias=False)
        self.bn3 = nn.BatchNorm2d(branch_channels)
        
        # Branch 4 (Dilation = 7)
        self.conv4 = nn.Conv2d(in_channels + 4 * branch_channels, branch_channels, kernel_size=3,
                               dilation=7, padding=7, bias=False)
        self.bn4 = nn.BatchNorm2d(branch_channels)
        
        # Branch 5 (Dilation = 13)
        self.conv5 = nn.Conv2d(in_channels + 5 * branch_channels, branch_channels, kernel_size=3,
                               dilation=13, padding=13, bias=False)
        self.bn5 = nn.BatchNorm2d(branch_channels)

        # ---- Final 1x1 Convolution ----
        # To consolidate all the features from the dense connections
        final_in_channels = in_channels + 6 * branch_channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # Store original spatial size
        h, w = x.shape[2], x.shape[3]

        # Global pooling branch
        y1_pool = self.global_pool_conv(x)
        y1_pool = self.leaky_relu(y1_pool)
        # Upsample back to original size
        y1 = F.interpolate(y1_pool, size=(h, w), mode='bilinear', align_corners=True)
        o1 = torch.cat([x, y1], dim=1)

        # Dilated branches
        y2 = self.leaky_relu(self.bn1(self.conv1(o1)))
        o2 = torch.cat([o1, y2], dim=1)

        y3 = self.leaky_relu(self.bn2(self.conv2(o2)))
        o3 = torch.cat([o2, y3], dim=1)
        
        y4 = self.leaky_relu(self.bn3(self.conv3(o3)))
        o4 = torch.cat([o3, y4], dim=1)
        
        y5 = self.leaky_relu(self.bn4(self.conv4(o4)))
        o5 = torch.cat([o4, y5], dim=1)
        
        y6 = self.leaky_relu(self.bn5(self.conv5(o5)))
        
        # Concatenate all features for the final convolution
        out = torch.cat([o5, y6], dim=1)
        
        # Final feature consolidation
        out = self.final_conv(out)
        return out

class AttentionUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, features=[64, 128, 256, 512, 1024]):
        super(AttentionUNet, self).__init__()

        # Downsampling/Encoder Path
        self.EncoderConv1 = ConvBlock(in_channels, features[0])
        self.EncoderConv2 = ConvBlock(features[0], features[1])
        self.EncoderConv3 = ConvBlock(features[1], features[2])
        self.EncoderConv4 = ConvBlock(features[2], features[3])
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.Bottleneck = DDSPP(in_channels=features[3], out_channels=features[4], branch_channels=256)


        # Upsampling/Decoder Path
        self.Up5 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.Att5 = AttentionGate(F_g=features[3], F_l=features[3], F_int=features[2])
        self.DecoderConv5 = ConvBlock(features[4], features[3])

        self.Up4 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.Att4 = AttentionGate(F_g=features[2], F_l=features[2], F_int=features[1])
        self.DecoderConv4 = ConvBlock(features[3], features[2])

        self.Up3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.Att3 = AttentionGate(F_g=features[1], F_l=features[1], F_int=features[0])
        self.DecoderConv3 = ConvBlock(features[2], features[1])

        self.Up2 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.Att2 = AttentionGate(F_g=features[0], F_l=features[0], F_int=32)
        self.DecoderConv2 = ConvBlock(features[1], features[0])

        # Final 1x1 Convolution to map to output channels
        self.FinalConv = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder path (The forward pass logic does not change at all)
        x1 = self.EncoderConv1(x)
        x2 = self.EncoderConv2(self.Maxpool(x1))
        x3 = self.EncoderConv3(self.Maxpool(x2))
        x4 = self.EncoderConv4(self.Maxpool(x3))
        x5 = self.Bottleneck(self.Maxpool(x4)) # <-- The new module is called here

        # Decoder path with attention gates
        d5 = self.Up5(x5)
        x4_att = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.DecoderConv5(d5)

        d4 = self.Up4(d5)
        x3_att = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.DecoderConv4(d4)

        d3 = self.Up3(d4)
        x2_att = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.DecoderConv3(d3)

        d2 = self.Up2(d3)
        x1_att = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.DecoderConv2(d2)

        out = self.FinalConv(d2)
        
        if self.FinalConv.out_channels == 1:
            return torch.sigmoid(out)
        else:
            return out