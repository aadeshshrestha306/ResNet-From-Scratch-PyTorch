import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """34 layer version of ResNet which uses basic block"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # 1st layer 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2nd layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()

        # reduces dimensionalty of the input after each residual block
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """Forward function for a basic block with shortcut addition"""
        identity = x # X

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) # F(X)

        out += self.shortcut(identity) # F(X) + X

        out = self.relu(out)

        return out 


class BottleNeck(nn.Module):
    """50 layer version of ResNet which uses bottleneck block"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()

        # 1st layer (Reduce Channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False) 
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 2nd layer (Process)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 3rd layer (Expand), the output channel/filters increases by 4 times
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != (out_channels * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
        
    def forward(self, x):
        """Forward function for a bottleneck block with shortcut addition"""
        identity = x # X

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) 
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out) # F(X)

        out += self.shortcut(identity) # F(X) + X

        out = self.relu(out)

        return out 

    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 16
        self.num_classes = num_classes

        # initial layer suited for CIFAR-10 dataset
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # stack of blocks
        self.layer_1 = self._make_layer(block, 16, layers[0], 1)
        self.layer_2 = self._make_layer(block, 32, layers[1], 2)
        self.layer_3 = self._make_layer(block, 64, layers[2], 2)
        self.layer_4 = self._make_layer(block, 64, layers[3], 2)
        
        # average pool and fully connected layer 
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_block, stride):
        """Function that creates the stack of Residual Blocks"""

        strides = [stride] + [1] * (num_block - 1)
        layers = []

        for current_stride in strides:
            layers.append(
                block(self.in_channels, out_channels, current_stride)
            )
            self.in_channels = out_channels * block.expansion

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
    
resnet34 = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=10)
resnet50 = ResNet(BottleNeck, [3, 4, 6, 3], num_classes=10)