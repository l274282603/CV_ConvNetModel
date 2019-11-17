import torch
import torch.nn as nn
from torchvision.models import squeezenet1_0
from torchvision import transforms
from PIL import Image


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()

        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, X):
        X = self.squeeze_activation(self.squeeze(X))
        X = torch.cat([
            self.expand1x1_activation(self.expand1x1(X)),
            self.expand3x3_activation(self.expand3x3(X))
        ], dim=1)

        return X



class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 1000, kernel_size=1),   #输出 13*13*1000
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  #输出 1*1*1000
        )

    def forward(self, X):
        X = self.features(X)
        print(X.shape)
        X = self.classifier(X)
        return torch.flatten(X, 1)

#对图像的预处理（固定尺寸到224， 转换成touch数据, 归一化）
tran = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    image = Image.open("tiger.jpeg")
    image = tran(image)
    image = torch.unsqueeze(image, dim=0)

    net = SqueezeNet()
    # net = squeezenet1_0()
    for name, parameter in net.named_parameters():
        print("name={},size={}".format(name, parameter.size()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    image = image.to(device)
    net.load_state_dict(torch.load("squeezenet1_0-a815701f.pth"))  # 加载pytorch中训练好的模型参数
    net.eval()

    output = net(image)
    test, prop = torch.max(output, 1)
    synset = [l.strip() for l in open("synset.txt").readlines()]
    print("top1：", synset[prop.item()])

    preb_index = torch.argsort(output, dim=1, descending=True)[0]
    top5 = [(synset[preb_index[i]], output[0][preb_index[i]].item()) for i in range(5)]
    print(("Top5: ", top5))



