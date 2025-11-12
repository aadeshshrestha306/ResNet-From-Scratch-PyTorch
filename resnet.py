import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """I am implementing the 34 layer version of ResNet which uses the basic block concept"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # 1st layer 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2nd layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()

        # reduces dimensionalty of the input after each residual block
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels)
            )

        
    def forward(self, x):
        """Forward function for a residual block with shortcut addition"""
        identity = x # X

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) # F(X)

        out += self.shortcut(identity) # F(X) + X

        out = self.relu(out)

        return out 
    

class ResNet34(nn.Module):
    """34 layer architecture with number of filters as per the paper"""
    
    def __init__(self, block, layers, num_classes):
        super(ResNet34, self).__init__()

        self.in_channels = 16
        self.num_classes = num_classes

        # initial layer suited for CIFAR 10 dataset
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # the real layers
        self.layer_1 = self._make_layer(block, 16, layers[0], 1)
        self.layer_2 = self._make_layer(block, 32, layers[1], 2)
        self.layer_3 = self._make_layer(block, 64, layers[2], 2)
        self.layer_4 = self._make_layer(block, 64, layers[3], 2)
        
        # average pool and fully connected layer 
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)


    def _make_layer(self, block, out_channels, num_block, stride):
        """Function that creates the stack of ResNet layers"""

        strides = [stride] + [1] * (num_block - 1)
        layers = []

        for current_stride in strides:
            layers.append(
                block(self.in_channels, out_channels, current_stride)
            )
            self.in_channels = out_channels

        return nn.Sequential(*layers)
    

    def forward(self, x):
        """Forward Pass for ResNet"""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer_1(out) 
        out = self.layer_2(out) 
        out = self.layer_3(out)
        out = self.layer_4(out) 

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
    
resnet34 = ResNet34(BasicBlock, [3, 4, 6, 3], num_classes=10)
