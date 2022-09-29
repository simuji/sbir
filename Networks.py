import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import torch



class Resnet50_Network(nn.Module):
    def __init__(self, hp):
        super(Resnet50_Network, self).__init__()
        backbone = backbone_.resnet50(pretrained=True) #resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
        self.pool_method =  nn.AdaptiveMaxPool2d(1)


    def forward(self, input, features=True):
        x = self.features(input)
        x = self.pool_method(x)
        x = torch.flatten(x,1)

        print(F.normalize(x).shape)
        return F.normalize(x)



class baseline(nn.Module):
    def __init__(self,hp):
        super(baseline, self).__init__()

        #specific_features
        self.base_photo=Resnet50_photo(hp)
        self.base_sketch=Resnet50_sketch(hp)
        #common features
        self.base_common=Resnet_common(hp)

    def forward(self,photo_p,photo_n,sketch,train=True):
        share_featrue_s=self.base_sketch(sketch)
        share_featrue_p=self.base_photo(photo_p)
        share_featrue_n=self.base_photo(photo_n)


        photo_p_common,photo_n_common,sketch_common=self.base_common(share_featrue_p,share_featrue_n,share_featrue_s)

        if train:
            return photo_p_common,photo_n_common,sketch_common
        else:
            return photo_p_common,sketch_common
class Resnet50_sketch(nn.Module):
    def __init__(self, hp):
        super(Resnet50_sketch, self).__init__()
        backbone = backbone_.resnet50(pretrained=True) #resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
                if name=='layer2':
                    break
        for name,module in backbone.named_children():
            if name=='layer3':
                self.layer3=module
            if name=='layer4':
                self.layer4=module
        self.pool_method =  nn.AdaptiveMaxPool2d(1)


    def forward(self, input, bb_box = None):
        x = self.features(input)
        #x = self.pool_method(x)
        #x = torch.flatten(x,1)
        return x

class Resnet50_photo(nn.Module):
    def __init__(self,hp):
        super(Resnet50_photo, self).__init__()
        backbone = backbone_.resnet50(pretrained=True)  # resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
                if name=='layer2':
                    break
        self.pool_method = nn.AdaptiveMaxPool2d(1)

    def forward(self, input, bb_box = None):
        x = self.features(input)
        #x = self.pool_method(x)
        #x = torch.flatten(x,1)
        return x

class Resnet_common(nn.Module):
    def __init__(self,hp):
        super(Resnet_common, self).__init__()
        backbone=backbone_.resnet50(pretrained=True)


        self.features=nn.Sequential()

        for name,module in backbone.named_children():
            if name in['layer3','layer4']:
                self.features.add_module(name,module)

        self.pool_method=nn.AdaptiveMaxPool2d(1)

    def forward(self,feat1,feat2,feat3):
        x=torch.cat([feat1,feat2,feat3])
        x=self.features(x)
        x=self.pool_method(x)
        x=torch.flatten(x,1)

        split=int(x.shape[0]/3)
        print(split)
        return F.normalize(x[0:split]),F.normalize(x[split:split*2]),F.normalize(x[split*2:split*3])


class VGG_Network(nn.Module):
    def __init__(self, hp):
        super(VGG_Network, self).__init__()
        self.backbone = backbone_.vgg16(pretrained=True).features
        self.pool_method =  nn.AdaptiveMaxPool2d(1)

    def forward(self, input, bb_box = None):
        x = self.backbone(input)
        x = self.pool_method(x).view(-1, 512)
        return F.normalize(x)





class InceptionV3_Network(nn.Module):
    def __init__(self, hp):
        super(InceptionV3_Network, self).__init__()
        backbone = backbone_.inception_v3(pretrained=True)

        ## Extract Inception Layers ##
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default


    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        backbone_tensor = self.Mixed_7c(x)
        feature = self.pool_method(backbone_tensor).view(-1, 2048)
        return F.normalize(feature)



