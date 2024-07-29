import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(ConformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.conv = nn.Conv1d(d_model, dim_feedforward, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x2 = self.norm(x)
        attn_output, _ = self.self_attn(x2, x2, x2)
        x = x + self.dropout(attn_output)
        
        x2 = x.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        x2 = self.relu(self.conv(x2))
        x2 = self.fc(x2.permute(2, 0, 1))
        x = x + self.dropout(x2)
        x = x.permute(1, 0, 2)
        
        return x

class audioEncoder(nn.Module):
    def __init__(self, layers, num_filters):
        super(audioEncoder, self).__init__()
        block = SEBasicBlock
        self.inplanes = num_filters[0]

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=7, stride=(2, 1), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        # Adjust this projection layer to match the expected d_model
        self.proj_layer = nn.Conv1d(num_filters[3], 128, kernel_size=1)
        
        self.d_model = 128
        self.nhead = 8
        self.dim_feedforward = 512
        self.conformer_layers = 4
        self.conformer = nn.ModuleList([ConformerLayer(self.d_model, self.nhead, self.dim_feedforward) for _ in range(self.conformer_layers)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pooling and projection to match d_model
        x = torch.mean(x, dim=2, keepdim=True)
        x = x.view((x.size()[0], x.size()[1], -1))
        x = x.transpose(1, 2)

        x = self.proj_layer(x.permute(0, 2, 1))
        x = x.transpose(1, 2)

        for layer in self.conformer:
            x = layer(x)

        return x
