import torch.nn as nn

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(out_channels // 16, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.upsample(x)
        
        return x
        

class SementicDecoder(nn.Module):
    def __init__(self, num_classes, p3_channel, p4_channel, p5_channel):
        """
        input:
            num_classes: number of classes
    
        model:
            FPN

        ref:
            | Layer         | Shape                        | Channel |
            | ------------- | ---------------------------- | ------- |
            | P3 (Layer 16) | torch.Size([1, 64, 80, 60])  | 64      |
            | P4 (Layer 19) | torch.Size([1, 128, 40, 30]) | 128     |
            | P5 (Layer 22) | torch.Size([1, 256, 20, 15]) | 256     |
        """

        super().__init__()
        self.num_classes = num_classes
        self.p3_up = nn.Sequential(
            nn.Conv2d(p3_channel, p3_channel, kernel_size=3, padding=1)
        )
        self.p4_up = nn.Sequential(
            UpsampleBlock(p4_channel, p3_channel)
        )
        self.p5_up = nn.Sequential(
            UpsampleBlock(p5_channel, p4_channel),
            UpsampleBlock(p4_channel, p3_channel)
        )

        self.out = nn.Sequential(
            nn.Conv2d(p3_channel, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        )

    def forward(self, p3, p4, p5):
        p3 = self.p3_up(p3)
        p4 = self.p4_up(p4)
        p5 = self.p5_up(p5)
        
        sum = p3 + p4 + p5

        out = self.out(sum)

        return out
        


        
        



