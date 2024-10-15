import math
import timm
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights



class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # One in and one out

    def forward(self, x):
        x = x['direction_distance'][:, 1:]
        y_pred = self.linear(x)
        return y_pred


class MLPRegressionModel(torch.nn.Module):
    def __init__(self):
        super(MLPRegressionModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 8), torch.nn.ReLU(),
            torch.nn.Linear(8, 8), torch.nn.ReLU(),
            torch.nn.Linear(8, 1), torch.nn.ReLU()
        )
    
    def forward(self, x):
        x = x['direction_distance'][:, 1:]
        y_pred = self.net(x)
        return y_pred


class DirDis(nn.Module):
    def __init__(self, layer_sizes=[2, 16, 16, 1], **kwargs):
        super(DirDis, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=False))
            if i < len(layer_sizes) - 2:  # Add ReLU activation between layers, but not after the last layer
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(nn.Tanh()) # NewGELU())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, inputs):
        x = inputs['direction_distance']
        x = self.network(x)
        return x
    

def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, use_relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=paddings)]
    if batch_norm:
        model.append(nn.BatchNorm2d(output_channels))
    if use_relu:
        model.append(nn.ReLU())
    return nn.Sequential(*model)


class VisualNet(nn.Module):
    def __init__(self, original_resnet, num_channel=3, freeze_weights=True):
        super(VisualNet, self).__init__()
        original_resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers)  # features before conv1x1
        self.freeze_weights = freeze_weights
        if self.freeze_weights:
            self.freeze_the_weights()

    def freeze_the_weights(self):
        for param in self.feature_extraction.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.feature_extraction(x)
        return x


class EgoVisDis(nn.Module):
    def __init__(
        self, 
        resnet_type='resnet50',
        freeze_resnets=False, 
        use_rgb=False,
        use_depth=False,
        no_mask=False,
        limited_fov=False,
        mean_pool_visual=False,
        use_rgbd=False,
        use_regression=True,
        num_bins=128,
        distance_encoder_dim=16,
        layer_sizes=[64, 8, 1],
        **kwargs
    ):
        super(EgoVisDis, self).__init__()
        self.resnet_type = resnet_type
        self.layer_sizes = layer_sizes
        self.use_rgbd = use_rgbd
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_visual = use_rgb or use_depth or use_rgbd
        self.mean_pool_visual = mean_pool_visual
        self.resolution_w = 256
        self.resolution_h = 256
        
        if self.resnet_type == 'resnet50':
            num_feat_channels = 2048
        else:
            num_feat_channels = 512

        if self.use_visual:
            if use_rgb:
                if self.resnet_type == 'resnet50':
                    self.rgb_net = VisualNet(torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT), 3, freeze_weights=freeze_resnets)
                else:
                    self.rgb_net = VisualNet(torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT), 3, freeze_weights=freeze_resnets)
            if use_depth:
                if self.resnet_type == 'resnet50':
                    self.depth_net = VisualNet(torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT), 3, freeze_weights=freeze_resnets)
                else:
                    self.depth_net = VisualNet(torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT), 3, freeze_weights=freeze_resnets)
            if use_rgbd:
                if self.resnet_type == 'resnet50':
                    self.rgbd_net = VisualNet(torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT), 6, freeze_weights=freeze_resnets)
                else:
                    self.rgbd_net = VisualNet(torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT), 6, freeze_weights=freeze_resnets)
            
            concat_size = num_feat_channels * sum([self.use_rgb, self.use_depth, self.use_rgbd])
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            
            if self.mean_pool_visual:
                self.conv1x1 = create_conv(concat_size, num_feat_channels, 1, 0)
                vis_feat_channels = num_feat_channels
            else:
                self.conv1x1 = create_conv(concat_size, 8, 1, 0)
                vis_feat_channels = 8
        self.use_regression = use_regression 
        self.distance_encoder = nn.Linear(1, distance_encoder_dim)

        self.layer_sizes = [vis_feat_channels + distance_encoder_dim] + layer_sizes
        # print(self.layer_sizes)
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1], bias=False))
            # layers.append(nn.BatchNorm1d(self.layer_sizes[i + 1]))
            if i < len(self.layer_sizes) - 2:  # Add ReLU activation between layers, but not after the last layer
                layers.append(NewGELU())
        self.predictor = nn.Sequential(*layers)

    def forward(self, inputs):
        batch_size = inputs['direction_distance'].shape[0]
        seq_len = 4
        distance_embed = self.distance_encoder(inputs['direction_distance'][:, 1:])
        visual_features = []
        rgb = inputs['rgb']
        depth = inputs['depth']
       
        if self.use_rgb:
            visual_features.append(self.rgb_net(rgb))
        if self.use_depth:
            visual_features.append(self.depth_net(depth))
        if self.use_rgbd:
            visual_features.append(self.rgbd_net(torch.cat([rgb, depth], dim=1)))
        if len(visual_features) != 0:
            # concatenate channel-wise
            concat_visual_features = torch.cat(visual_features, dim=1)
            concat_visual_features = self.conv1x1(concat_visual_features)
        else:
            concat_visual_features = None

        if self.mean_pool_visual:
            concat_visual_features = self.pooling(concat_visual_features)
        elif len(visual_features) != 0:
            concat_visual_features = concat_visual_features.view(concat_visual_features.shape[0], -1, 1, 1)

        if len(visual_features) != 0:
            visual_embed = concat_visual_features.squeeze(-1).squeeze(-1)
            visual_feat = F.normalize(visual_embed, p=2, dim=1)

        visdirdis_feat = torch.concat([visual_feat, distance_embed], axis=1)
        return self.predictor(visdirdis_feat)


class EgoVisDisPool(EgoVisDis):
    def __init__(self, resnet_type='resnet50', freeze_resnets=False, use_rgb=False, use_depth=False, no_mask=False, limited_fov=False, mean_pool_visual=False, use_rgbd=False, use_regression=True, num_bins=128, distance_encoder_dim=16, layer_sizes=[64, 8, 1], **kwargs):
        super().__init__(resnet_type, freeze_resnets, use_rgb, use_depth, no_mask, limited_fov, mean_pool_visual, use_rgbd, use_regression, num_bins, distance_encoder_dim, layer_sizes, **kwargs)
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1], bias=False))
            layers.append(nn.AdaptiveAvgPool1d(self.layer_sizes/2))
            # layers.append(nn.BatchNorm1d(self.layer_sizes[i + 1]))
            if i < len(self.layer_sizes) - 2:  # Add ReLU activation between layers, but not after the last layer
                layers.append(NewGELU())
        self.predictor = nn.Sequential(*layers)

    def forward(self, inputs):
        return super().forward(inputs)


class VisDirDis(nn.Module):
    def __init__(
        self, 
        freeze_resnets=False, 
        use_rgb=False,
        use_depth=False,
        no_mask=False,
        limited_fov=False,
        mean_pool_visual=False,
        use_rgbd=False,
        use_regression=True,
        num_bins=128,
        layer_sizes=[528, 256, 64, 8, 1],
        dirdis_embed_dim = 16,
        add_preactivation_batchnorm=False,
        **kwargs
    ):
        super(VisDirDis, self).__init__()
        self.dirdis_embed_dim = dirdis_embed_dim
        self.layer_sizes = layer_sizes
        self.use_rgbd = use_rgbd
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_visual = use_rgb or use_depth or use_rgbd
        self.mean_pool_visual = mean_pool_visual
        self.resolution_w = 256
        self.resolution_h = 256
        self.freeze_resnets = freeze_resnets
        
        if self.use_visual:
            if use_rgb:
                self.rgb_net = VisualNet(torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT), 3, freeze_weights=self.freeze_resnets)
            if use_depth:
                self.depth_net = VisualNet(torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT), 3, freeze_weights=self.freeze_resnets)
            if use_rgbd:
                self.rgbd_net = VisualNet(torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT), 6, freeze_weights=self.freeze_resnets)
            concat_size = 512 * sum([self.use_rgb, self.use_depth, self.use_rgbd])
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            if self.mean_pool_visual:
                self.conv1x1 = create_conv(concat_size, 512, 1, 0)
            else:
                self.conv1x1 = create_conv(concat_size, 8, 1, 0)
        self.use_regression = use_regression
        if use_regression:
            out_features = 1
        else:
            out_features = num_bins
        
        self.direction_distance_encoder = nn.Linear(2, self.dirdis_embed_dim)
        # concat_size = 2 + 512 if self.mean_pool_visual else 8 

        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1], bias=False))
            if i < len(self.layer_sizes) - 2:  # Add ReLU activation between layers, but not after the last layer
                if add_preactivation_batchnorm:
                    layers.append(nn.BatchNorm1d(self.layer_sizes[i + 1]))
                layers.append(NewGELU())
        
        self.predictor = nn.Sequential(*layers)

    def forward(self, inputs):
        batch_size = inputs['direction_distance'].shape[0]
        seq_len = 4
        direction_distance_embed = self.direction_distance_encoder(inputs['direction_distance'])
        visual_features = []
        rgb = inputs['rgb']
        depth = inputs['depth']
       
        if self.use_rgb:
            visual_features.append(self.rgb_net(rgb))
        if self.use_depth:
            visual_features.append(self.depth_net(depth))
        if self.use_rgbd:
            visual_features.append(self.rgbd_net(torch.cat([rgb, depth], dim=1)))
        if len(visual_features) != 0:
            # concatenate channel-wise
            concat_visual_features = torch.cat(visual_features, dim=1)
            concat_visual_features = self.conv1x1(concat_visual_features)
        else:
            concat_visual_features = None

        if self.mean_pool_visual:
            concat_visual_features = self.pooling(concat_visual_features)
        elif len(visual_features) != 0:
            concat_visual_features = concat_visual_features.view(concat_visual_features.shape[0], -1, 1, 1)

        if len(visual_features) != 0:
            visual_embed = concat_visual_features.squeeze(-1).squeeze(-1)
            visual_feat = F.normalize(visual_embed, p=2, dim=1)

        visdirdis_feat = torch.concat([visual_feat, direction_distance_embed], axis=1)
        return self.predictor(visdirdis_feat)
    

class Resnet101VisDirDis(VisDirDis):
    def __init__(
        self, 
        freeze_resnets=False, 
        use_rgb=False,
        use_depth=False,
        no_mask=False,
        limited_fov=False,
        mean_pool_visual=False,
        use_rgbd=False,
        use_regression=True,
        num_bins=128,
        layer_sizes=[528, 256, 64, 8, 1],
        dirdis_embed_dim = 16,
        add_preactivation_batchnorm=False,
        **kwargs
    ):
        super(Resnet101VisDirDis, self).__init__(**kwargs)
        self.dirdis_embed_dim = dirdis_embed_dim
        self.layer_sizes = layer_sizes
        self.use_rgbd = use_rgbd
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_visual = use_rgb or use_depth or use_rgbd
        self.mean_pool_visual = mean_pool_visual
        self.resolution_w = 256
        self.resolution_h = 256
        self.freeze_resnets = freeze_resnets
        
        if self.use_visual:
            if use_rgb:
                self.rgb_net = VisualNet(torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT), 3, freeze_weights=self.freeze_resnets)
            if use_depth:
                self.depth_net = VisualNet(torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT), 3, freeze_weights=self.freeze_resnets)
            if use_rgbd:
                self.rgbd_net = VisualNet(torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT), 6, freeze_weights=self.freeze_resnets)
            concat_size = 2048 * sum([self.use_rgb, self.use_depth, self.use_rgbd])
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            if self.mean_pool_visual:
                self.conv1x1 = create_conv(concat_size, 2048, 1, 0)
            else:
                self.conv1x1 = create_conv(concat_size, 8, 1, 0)
        self.use_regression = use_regression
        if use_regression:
            out_features = 1
        else:
            out_features = num_bins
        
        self.direction_distance_encoder = nn.Linear(2, self.dirdis_embed_dim)
        # concat_size = 2 + 512 if self.mean_pool_visual else 8 

        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1], bias=False))
            if i < len(self.layer_sizes) - 2:  # Add ReLU activation between layers, but not after the last layer
                if add_preactivation_batchnorm:
                    layers.append(nn.BatchNorm1d(self.layer_sizes[i + 1]))
                layers.append(NewGELU())
        
        self.predictor = nn.Sequential(*layers)


# class ViTVisDirDis(nn.Module):
#     def __init__(
#         self, 
#         mean_pool_visual=False,
#         use_rgbd=False,
#         use_regression=True,
#         num_bins=128,
#         layer_sizes=[528, 256, 64, 8, 1],
#         dirdis_embed_dim = 16,
#         add_preactivation_batchnorm=False,
#         **kwargs
#     ):
#         super(ViTVisDirDis, self).__init__()
#         self.dirdis_embed_dim = dirdis_embed_dim
#         self.layer_sizes = layer_sizes
#         self.use_rgbd = use_rgbd
#         self.use_regression = use_regression
#         self.mean_pool_visual = mean_pool_visual
#         self.resolution_w = 256
#         self.resolution_h = 256
        
#         if self.use_rgbd:
#             self.rgbd_net = VisualNet(torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT), 6, freeze_weights=False)
#             concat_size = 512 * sum([self.use_rgbd])
#             self.pooling = nn.AdaptiveAvgPool2d((1, 1))
#             if self.mean_pool_visual:
#                 self.conv1x1 = create_conv(concat_size, 512, 1, 0)
#             else:
#                 self.conv1x1 = create_conv(concat_size, 8, 1, 0)
        
#         if use_regression:
#             out_features = 1
#         else:
#             out_features = num_bins
        
#         self.direction_distance_encoder = nn.Linear(2, self.dirdis_embed_dim)
#         # concat_size = 2 + 512 if self.mean_pool_visual else 8 

#         layers = []
#         for i in range(len(self.layer_sizes) - 1):
#             layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1], bias=False))
#             if i < len(self.layer_sizes) - 2:


class ANP(nn.Module):
    def __init__(
        self, 
        input_channel=2, 
        use_rgb=False,
        use_depth=False,
        no_mask=False,
        limited_fov=False,
        mean_pool_visual=False,
        use_rgbd=False,
        use_regression=True,
        num_bins=128,
        **kwargs
    ):
        super(ANP, self).__init__()
        self.use_rgbd = use_rgbd
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_visual = use_rgb or use_depth or use_rgbd
        self.mean_pool_visual = mean_pool_visual
        self.resolution_w = 256
        self.resolution_h = 256
        
        if self.use_visual:
            if use_rgb:
                self.rgb_net = VisualNet(torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT), 3)
            if use_depth:
                self.depth_net = VisualNet(torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT), 3)
            if use_rgbd:
                self.rgbd_net = VisualNet(torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT), 4)
            concat_size = 512 * sum([self.use_rgb, self.use_depth, self.use_rgbd])
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            if self.mean_pool_visual:
                self.conv1x1 = create_conv(concat_size, 512, 1, 0)
            else:
                self.conv1x1 = create_conv(concat_size, 8, 1, 0)
        self.use_regression = use_regression
        if use_regression:
            out_features = 1
        else:
            out_features = num_bins
        
        self.direction_distance_encoder = nn.Linear(2, 16)
        # vis_decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=2)
        # self.visual_direction_distance_encoder = nn.TransformerDecoder(vis_decoder_layer, num_layers=4)
        # audio_decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=4)
        # self.ir_decoder = nn.TransformerDecoder(audio_decoder_layer, num_layers=6)
        if use_regression:
            self.predictor = nn.Sequential(
                nn.Linear(1024, 256), 
                nn.ReLU(),
                nn.Linear(256, out_features),
                nn.Sigmoid()
            )
        else:
            self.predictor = nn.Sequential(
                nn.Linear(1024, 256), 
                nn.ReLU(),
                nn.Linear(256, out_features),
            )
        # if complex, keep input output channel as 2
        # self.audio_net = AudioNet(64, input_channel, input_channel, self.use_visual, no_mask,
        #                           limited_fov, mean_pool_visual)
        # self.audio_net.apply(weights_init)

    def forward(self, inputs):
        batch_size = inputs['direction_distance'].shape[0]
        seq_len = 4
        direction_distance_embed = self.direction_distance_encoder(inputs['direction_distance'])
        visual_features = []
        rgb = inputs['rgb']
        depth = inputs['depth']
       
        if self.use_rgb:
            visual_features.append(self.rgb_net(rgb))
        if self.use_depth:
            visual_features.append(self.depth_net(depth))
        if self.use_rgbd:
            visual_features.append(self.rgbd_net(torch.cat([rgb, depth], dim=1)))
        if len(visual_features) != 0:
            # concatenate channel-wise
            concat_visual_features = torch.cat(visual_features, dim=1)
            concat_visual_features = self.conv1x1(concat_visual_features)
        else:
            concat_visual_features = None

        if self.mean_pool_visual:
            concat_visual_features = self.pooling(concat_visual_features)
        elif len(visual_features) != 0:
            concat_visual_features = concat_visual_features.view(concat_visual_features.shape[0], -1, 1, 1)

        if len(visual_features) != 0:
            visual_embed = concat_visual_features.squeeze(-1).squeeze(-1)
            visual_feat = F.normalize(visual_embed, p=2, dim=1)

        visdirdis_feat = torch.concat([visual_feat, direction_distance_embed], axis=1)
        # per batch features 
        # visual_feat = visual_feat.reshape(batch_size, 1, -1)
        # visual_feat = visual_feat.reshape(batch_size, -1)
        # # concat and predict
        # dirdis_embed = direction_distance_embed.permute(1, 0, 2)
        # visual_feat = visual_feat.permute(1, 0, 2)
        # vis_dist_feat = self.visual_direction_distance_encoder(tgt=dirdis_embed, memory=visual_feat).squeeze()
        # cross attend with distance and direction
        return self.predictor(visdirdis_feat)
        

class Heuristic(nn.Module):
    def __init__(self, initial_sound_level=120.0, **kwargs):
        super(Heuristic, self).__init__()
        self.initial_sound_level = initial_sound_level

    def forward(self, x):
        x = x['direction_distance']
        # y_pred = self.initial_sound_level - 20 * torch.log10(x[:, 1] * 10) 
        y_pred = 1. - 20 * torch.log10(x['distances']*10) / self.initial_sound_level 
        # y_pred /= self.initial_sound_level
        return y_pred.reshape(-1, 1)


class ViTDirDis(nn.Module):
    def __init__(self, modelname='vit_base_patch16_224', **kwargs):
        self.model = timm.create_model(modelname, pretrained=True)

    def forward(self, x):
        x = x['direction_distance']
        x = self.model.forward_features(x)
        return self.model(x)

if __name__=="__main__":
    from torch.utils.data import DataLoader
    from anp.data import AudioDecibelDataset, make_data_config

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    config = make_data_config('train', '/data/vdj/ss/anp_data_full-10/')
    dataset = AudioDecibelDataset(config)
    print(len(dataset))
    print(dataset[0])

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    entry = next(iter(dataloader))

    vnet = ANP(use_rgb=True, use_depth=True, mean_pool_visual=True, use_regression=False) #, use_depth=True)
    for key in entry:
        entry[key] = entry[key].to(device)
    vnet = vnet.to(device)
    out = vnet(entry)

    breakpoint()
    print("Done")
    