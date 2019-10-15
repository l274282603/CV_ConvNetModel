import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.models import vgg16
from torchvision import transforms
from PIL import Image

cfgs =  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def make_layers():
    layers = []
    in_channel = 3
    for cfg in cfgs:
        if cfg == 'M':
            layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        else:
            conv2d = nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channel = cfg
    return nn.Sequential(*layers)


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.features = make_layers()   #创建卷积层
        self.classifier = nn.Sequential(   #创建全连接层
            nn.Linear(in_features=512 * 7 *7, out_features = 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )


    def forward(self, input):   #前向传播过程
        feature = self.features(input)
        linear_input = torch.flatten(feature, 1)
        out_put = self.classifier(linear_input)
        return out_put

#对图像的预处理（固定尺寸到224， 转换成touch数据, 归一化）
tran = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])



if __name__ == '__main__':
    # image = cv2.imread("car.jpg")
    # image = cv2.resize(src=image, dsize=(224, 224))
    # image = image.reshape(1, 224, 224, 3)
    # image = torch.from_numpy(image)
    # image = image.type(torch.FloatTensor)
    # image = image.permute(0,3,1,2)

    image = Image.open("tiger.jpeg")
    image = tran(image)
    image = torch.unsqueeze(image, dim=0)

    net = VGGNet()
    net = net.to(device)
    image = image.to(device)
    net.load_state_dict(torch.load("vgg16-397923af.pth"))  #加载pytorch中训练好的模型参数
    net.eval()

    output = net(image)
    test, prop = torch.max(output, 1)
    synset = [l.strip() for l in open("synset.txt").readlines()]
    print("top1：",synset[prop.item()])

    preb_index = torch.argsort(output, dim=1, descending=True)[0]
    top5 = [(synset[preb_index[i]], output[0][preb_index[i]].item()) for i in range(5)]
    print(("Top5: ", top5))


