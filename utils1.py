import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.PReLU(in_planes // ratio)
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = avg_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(max_out)
        return self.sigmoid(x)


class BlancedAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(BlancedAttention, self).__init__()

        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        ca_ch = self.ca(x)
        sa_ch = self.sa(x)
        out = ca_ch.mul(sa_ch)*x
        return out


class BasicConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Shrinkage(nn.Module):
    def __init__(self,  channel):
        super(Shrinkage, self).__init__()
        # self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            # nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = x. squeeze()
        # average = torch.mean(x, dim=1, keepdim=True)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(-1)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)

        return x

class MFPM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MFPM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv1d(in_channel, out_channel, 1)
        )

        self.branch1 = nn.Sequential(
            BasicConv1d(in_channel, out_channel, 1),
            BasicConv1d(out_channel, out_channel, kernel_size=3, padding=1),
            BasicConv1d(out_channel, out_channel, 3, padding=3, dilation=3)
        )

        self.branch2 = nn.Sequential(
            BasicConv1d(out_channel, out_channel, 1),
            BasicConv1d(out_channel, out_channel, kernel_size=5, padding=2),
            BasicConv1d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.branch3 = nn.Sequential(
            BasicConv1d(out_channel, out_channel, 1),
            BasicConv1d(out_channel, out_channel, kernel_size=7, padding=3),
            BasicConv1d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.branch4 = nn.Sequential(
            BasicConv1d(out_channel, out_channel, 1),
            BasicConv1d(out_channel, out_channel, 9, padding=4),
            BasicConv1d(out_channel, out_channel, 3, padding=9, dilation=9)
        )

        self.conv = nn.Conv1d(in_channel, out_channel, 1)

        self.conv_cat = nn.Conv1d(out_channel*4, out_channel, 3, padding=1)
        # self.conv_cat = nn.Conv1d(out_channel*3, out_channel, 3, padding=1)
        self.shrinkage = Shrinkage(out_channel)
        self.shrinkage1 = Shrinkage(out_channel)

        self.conv_res = BasicConv1d(in_channel, out_channel, 1)
        self.shrinkage = Shrinkage(out_channel)

        self.fuse = BasicConv1d(2* out_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(self.conv(x) + x1)
        x3 = self.branch3(self.conv(x) + x2)
        x4 = self.branch4(self.conv(x) + x3)
        x_cat = self.conv_cat(torch.cat((x1, x2, x3, x4), dim=1))




        # x_cat = self.conv_cat(torch.cat((x1, x2, x3), dim=1))

        x11 = self.relu(x0 + x_cat)

        con = self.shrinkage(x11)
        res1 = self.conv_res(x)
        x1_add = torch.add(con, res1)
        x1_add = self.relu(x1_add)
        return x1_add
