import torch.nn as nn
import torch
import torchsummary
# torch version = 1.2.0
class baseConv(nn.Module):
    'the basic conv_bn_relu block'
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,relu=True) -> None:
        super().__init__()
        if relu:
            self.con = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:# else remove the ReLU part
            self.con = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True),
            )
    def forward(self, x):
        return self.con(x)

class ResBlock(nn.Module):
    'residual block, consists of 3x3,3x3 baseConv parts'
    exp = 1
    def __init__(self,in_channels,out_channels,stride=1) -> None:
        super().__init__()
        self.con = nn.Sequential(
            baseConv(in_channels,out_channels,kernel_size=3,stride=stride,padding=1),
            baseConv(out_channels,int(out_channels*self.exp),kernel_size=3,padding=1,relu=False),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels*self.exp: # ensure that following shortcut and out could be added
            self.shortcut = baseConv(in_channels,int(out_channels*self.exp),kernel_size=1,stride=stride,relu=False)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.con(x)
        return nn.ReLU(inplace=True)(shortcut+out)

class bottleNeck(nn.Module):
    'bottleNeck block, consists of 1x1,3x3,1x1 baseConv parts'
    exp = 4
    def __init__(self,in_channels,out_channels,stride=1) -> None:
        super().__init__()
        self.con = nn.Sequential(
            baseConv(in_channels,out_channels,kernel_size=1),
            baseConv(out_channels,out_channels,kernel_size=3,stride=stride,padding=1),
            baseConv(out_channels,int(out_channels*self.exp),kernel_size=1,relu=False),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels*self.exp: # ensure that following shortcut and out could be added
            self.shortcut = baseConv(in_channels,int(out_channels*self.exp),kernel_size=1,stride=stride,relu=False)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.con(x)
        return nn.ReLU(inplace=True)(out+shortcut)

class ResNet(nn.Module):
    inplanes  = 64
    def _makeStage(self,block,num_blocks,out_channels,stride):# block could be ResBlock and bottleNeck,
        strides = [stride] + [1] * (num_blocks - 1) # for example, if stride=2 and num_blocks=3, strides will be [2,1,1], which contains the stride of each block
        stages = [] # ResNet has 4 stages, every stage contains some blocks, for example, if block=bottleNeck and num_blocks=3, the stages will be 3 bottleNect
        for i in strides:
            stages.append(block(self.inplanes,out_channels,stride=i))
            self.inplanes = int(block.exp * out_channels)# refresh the inplanes
        return nn.Sequential(*stages)

    def __init__(self,block,num_blocks,num_classes=10) -> None: # you can change the num_classes for your own datasets
        super().__init__()
        self.stem = nn.Sequential(# "stem" means the first part, construct the first 7x7 conv and 2x2 maxpool, which contains conv1_x and the first pool part in conv2_x
            baseConv(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.stage1 = self._makeStage(block,num_blocks[0],out_channels=64,stride=1)# this is conv2_x in the paper, so stride is 1
        self.stage2 = self._makeStage(block,num_blocks[1],out_channels=128,stride=2)# this is conv3_x in the paper,
        self.stage3 = self._makeStage(block,num_blocks[2],out_channels=256,stride=2)# this is conv4_x in the paper,
        self.stage4 = self._makeStage(block,num_blocks[3],out_channels=512,stride=2)# this is conv5_x in the paper,
        self.stages = nn.Sequential(
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(block.exp*512),num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

def resnet18():
    return ResNet(ResBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(ResBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(bottleNeck, [3, 4, 6, 3])

def resnet101():
    return ResNet(bottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(bottleNeck, [3, 8, 36, 3])

if __name__ == "__main__":
    model = resnet50()
    print(model)
    # if you have GPU, You can use the following code to show this net
    # device = torch.device('cuda:0')
    # model.to(device=device)
    # torchsummary.summary(model,input_size=(3,416,416))


