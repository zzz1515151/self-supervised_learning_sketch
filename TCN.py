import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class TCN(nn.Module):
    def __init__(self, opt):
        super(TCN, self).__init__()
        self.is_feature_extractor = opt["is_feature_extractor"]
        self.convs = nn.ModuleList([
            nn.Conv2d(1, opt["num_filters"], [window_size, 4], padding=(window_size - 1, 0))
            for window_size in opt["window_sizes"]
            ])

        #self.bn = nn.BatchNorm1d(opt["num_filters"] * len(opt["window_sizes"]))
        self.fc1 = nn.Linear(opt["num_filters"] * len(opt["window_sizes"]), \
                   4096)       
        self.bn = nn.BatchNorm1d(4096) 
        self.fc2 = nn.Linear(4096, opt["num_classes"])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)            
        x = torch.cat(xs, 2) 

        x = x.view(x.size(0), -1)        
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        if not self.is_feature_extractor:
            logits = self.fc2(x)
            return logits
        else:
            return x


