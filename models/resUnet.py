import torch
import torch.nn as nn

class ResConv_Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResConv_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x2 = self.conv(x)
        x1 = self.conv_skip(x)
        x = x1 + x2
        return x

class UPsam_Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UPsam_Block, self).__init__()
        self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x

class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pooling(x)
        return x
    
    import torch
import torch.nn as nn

class XAttention(nn.Module):  # H*H
    def __init__(self, ch_in, ch_out):
        super(XAttention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.conv1(x)
        q = q.permute(0, 2, 1, 3).reshape(b, h, -1)

        k = self.conv2(x)
        k = k.permute(0, 2, 1, 3).reshape(b, h, -1).permute(0, 2, 1)

        out = torch.matmul(q, k)  # H*H
        out = self.softmax(out)

        v = self.conv3(x)
        v = v.permute(0, 2, 1, 3).reshape(b, h, -1)

        out = torch.matmul(out, v)  # 1,H,C*W
        out = self.softmax(out)

        out = out.reshape(b, h, c, w).permute(0, 2, 1, 3)
        out = self.conv4(out)
        return out

# 沿y方向
class YAttention(nn.Module):  # W*W
    def __init__(self, ch_in, ch_out):
        super(YAttention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.conv1(x)
        q = q.permute(0, 3, 1, 2).reshape(b, w, -1)

        k = self.conv2(x)
        k = k.permute(0, 3, 1, 2).reshape(b, w, -1).permute(0, 2, 1)

        out = torch.matmul(q, k)  # W*W
        out = self.softmax(out)

        v = self.conv3(x)
        v = v.permute(0, 3, 1, 2).reshape(b, w, -1)

        out = torch.matmul(out, v)  # 1,W,C*H
        out = self.softmax(out)

        out = out.reshape(b, w, c, h).permute(0, 2, 3, 1)
        out = self.conv4(out)
        return out

class OAFF(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(OAFF, self).__init__()
        self.x_blanch = XAttention(ch_in, ch_out)
        self.y_blanch = YAttention(ch_in, ch_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        xblanch = self.x_blanch(x)
        yblanch = self.y_blanch(x)
        out = torch.cat((xblanch, x, yblanch), dim=1)  # x方向，y方向，以及本身，三个维度结合
        out = self.conv1(out)
        return out

class self_net(nn.Module):
    def __init__(self, ch_in=3, num_classes=4):
        super(self_net, self).__init__()

        self.down1 = ResConv_Block(ch_in, ch_out=64)

        # 下采样 + 残差块
        self.down2 = nn.Sequential(
            Pool(),
            ResConv_Block(ch_in=64, ch_out=128)
        )
        self.down3 = nn.Sequential(
            Pool(),
            ResConv_Block(ch_in=128, ch_out=256)
        )
        self.down4 = nn.Sequential(
            Pool(),
            ResConv_Block(ch_in=256, ch_out=512)
        )
        self.down5 = nn.Sequential(
            Pool(),
            ResConv_Block(ch_in=512, ch_out=1024)
        )
        
        
#         跳跃连接
        self.da4=OAFF(ch_in=512,ch_out=512)
        self.da3=OAFF(ch_in=256,ch_out=256)
    
        self.da2=OAFF(ch_in=128,ch_out=128)
    
        self.da1=OAFF(ch_in=64,ch_out=64)
    

        # 上采样 + 残差块
        self.up1 = UPsam_Block(ch_in=1024, ch_out=512)
        self.res1 = ResConv_Block(ch_in=1024, ch_out=512)

        self.up2 = UPsam_Block(ch_in=512, ch_out=256)
        self.res2 = ResConv_Block(ch_in=512, ch_out=256)

        self.up3 = UPsam_Block(ch_in=256, ch_out=128)
        self.res3 = ResConv_Block(ch_in=256, ch_out=128)

        self.up4 = UPsam_Block(ch_in=128, ch_out=64)
        self.res4 = ResConv_Block(ch_in=128, ch_out=64)

        self.outconv = nn.Conv2d(64, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.down1(x)
        # print(x1.shape)
        x2 = self.down2(x1)
        # print(x2.shape)
        x3 = self.down3(x2)
        # print(x3.shape)
        x4 = self.down4(x3)
        # print(x4.shape)
        x5 = self.down5(x4)

        y = self.up1(x5)
        x4=self.da4(x4)
        y1 = torch.cat((x4, y), dim=1)#跳跃连接
        # print(y1.shape)
        y2 = self.res1(y1)
        y3 = self.up2(y2)
        x3=self.da3(x3)
        y4 = torch.cat((x3, y3), dim=1)
        y5 = self.res2(y4)

        y6 = self.up3(y5)
        x2=self.da2(x2)
        y7 = torch.cat((x2, y6), dim=1)
        y8 = self.res3(y7)

        y9 = self.up4(y8)
        x1=self.da1(x1)
        y10 = torch.cat((x1, y9), dim=1)
        y11 = self.res4(y10)

        x6 = self.outconv(y11)
        # print("Final output:", x6.shape)  # 打印最终输出的尺寸 

        return x6
    
    
torch.save(self_net().state_dict(), 'model.pth')  
    

# # 创建一个随机输入张量，假设输入图像大小为256x256，通道数为3  
# input_tensor = torch.randn(1, 3, 224, 224)  # 批量大小1，通道数3，高度256，宽度256  
 
# # 创建ResUnet实例  
# model = self_net()  
 
# # 调用forward方法  
# output = model(input_tensor)  
 
# # 打印输出形状  
# # print(output.shape)