import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet

from mmcv.ops import DeformConv2d
from model_detection.modules.recurrent_module import ConvRNN
# from model_detection.modules.cmt_module import CMTB
from model_detection.modules.cbam import ChannelGate, SpatialGate
from model_detection.modules.adaptive_instance_normalization import adaptive_instance_normalization as adain


class Conv_BN_ReLU(nn.Module):
    def __init__(self, input_channels, output_channles,
                 kernel_size=3, stride=1, padding=1,
                 bias=True):
        super().__init__()
        self.CBR = nn.Sequential(
            nn.Conv2d(input_channels, output_channles,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(output_channles),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.CBR(x)


## BottleNeck with group
class GroupBottleNeck(nn.Module):
    def __init__(self, input_channels, middle_channels, output_channels,
                 kernel_size, padding, stride, groups, dilation, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, middle_channels, kernel_size=1, stride=1,
                               groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride,
                               groups=groups, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, output_channels, kernel_size=1, stride=1,
                               groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Encoder_d8(nn.Module):
    def __init__(self, channel_list):
        super().__init__()
        assert len(channel_list) == 4
        assert channel_list[1] % 4 == 0
        assert channel_list[2] % 4 == 0
        assert channel_list[3] % 4 == 0

        for i in range(len(channel_list)-1):
            assert channel_list[i] <= channel_list[i + 1], \
                '{}, {}'.format(channel_list[i], channel_list[i + 1])

        # self.input_conv = nn.Conv2d(channel_list[0], channel_list[1], kernel_size=5, stride=1, padding=2, bias=True)
        self.input_conv = Conv_BN_ReLU(channel_list[0], channel_list[1],
                                       kernel_size=5, stride=1, padding=2,
                                       bias=True)

        self.conv_1 = nn.Sequential(
            GroupBottleNeck(channel_list[1], channel_list[1] // 4, channel_list[1],
                            kernel_size=3, padding=1, stride=2,
                            groups=1, dilation=1,
                            downsample=nn.Conv2d(channel_list[1], channel_list[1], kernel_size=1, stride=2, groups=1)
                            ),
            GroupBottleNeck(channel_list[1], channel_list[1] // 4, channel_list[1],
                            kernel_size=3, padding=1, stride=1,
                            groups=1, dilation=1,
                            downsample=None
                            ),
            GroupBottleNeck(channel_list[1], channel_list[1] // 4, channel_list[1],
                            kernel_size=3, padding=1, stride=1,
                            groups=1, dilation=1,
                            downsample=None
                            )
        )

        self.conv_2 = nn.Sequential(
            GroupBottleNeck(channel_list[1], channel_list[1] // 4, channel_list[2],
                            kernel_size=3, padding=1, stride=2,
                            groups=1, dilation=1,
                            downsample=nn.Conv2d(channel_list[1], channel_list[2], kernel_size=1, stride=2, groups=1)
                            ),
            GroupBottleNeck(channel_list[2], channel_list[2] // 4, channel_list[2],
                            kernel_size=3, padding=1, stride=1,
                            groups=1, dilation=1,
                            downsample=None
                            ),
            GroupBottleNeck(channel_list[2], channel_list[2] // 4, channel_list[2],
                            kernel_size=3, padding=1, stride=1,
                            groups=1, dilation=1,
                            downsample=None
                            )
        )

        self.conv_3 = nn.Sequential(
            GroupBottleNeck(channel_list[2], channel_list[2] // 4, channel_list[3],
                            kernel_size=3, padding=1, stride=2,
                            groups=1, dilation=1,
                            downsample=nn.Conv2d(channel_list[2], channel_list[3], kernel_size=1, stride=2, groups=1)
                            ),
            GroupBottleNeck(channel_list[3], channel_list[3] // 4, channel_list[3],
                            kernel_size=3, padding=1, stride=1,
                            groups=1, dilation=1,
                            downsample=None
                            )
        )

    def forward(self, x):
        x_1 = self.conv_1(self.input_conv(x))
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)
        return x_1, x_2, x_3



## Simple mode
class ModalityWeightingModule(nn.Module):
    def __init__(self, frame_input_channels, event_input_channels,
                 frame_output_channels, event_output_channels):
        super().__init__()

        self.frame_c = frame_output_channels
        self.event_c = event_output_channels

        self.frame_CBR = Conv_BN_ReLU(frame_input_channels, self.frame_c,
                                      kernel_size=3, stride=1, padding=1)
        self.event_CBR = Conv_BN_ReLU(event_input_channels, self.event_c,
                                      kernel_size=3, stride=1, padding=1)


    def forward(self, frame_coarse_feature, event_coarse_feature):
        frame_feature = self.frame_CBR(frame_coarse_feature)  # [B, C, H, W]
        event_feature = self.event_CBR(event_coarse_feature)  # [B, C, H, W]

        frame_feature_weighted = frame_feature + frame_feature*event_feature
        event_feature_weighted = event_feature + event_feature*frame_feature

        return frame_feature_weighted, event_feature_weighted





class FusionBlock(nn.Module):
    def __init__(self, frame_input_channels, event_input_channels,
                 output_channels,
                 shared_channels,
                 last_feature_channels=None):
        super().__init__()
        self.frame_c = shared_channels
        self.event_c = shared_channels
        self.last_feature_channels = last_feature_channels

        ## 1. Modality weighting
        self.modality_weighting = ModalityWeightingModule(frame_input_channels, event_input_channels,
                                                          self.frame_c, self.event_c)

        ## 2. Channel attension
        self.frame_CA = ChannelGate(self.frame_c)
        self.event_CA = ChannelGate(self.event_c)


        ## 3. spatial attension
        self.frame_SA = SpatialGate()
        self.event_SA = SpatialGate()


        if self.last_feature_channels is None:
            self.output_BCR = Conv_BN_ReLU(self.frame_c+self.event_c, output_channels, 3, 1, 1)
        else:
            self.output_BCR = Conv_BN_ReLU((self.frame_c + self.event_c)+last_feature_channels, output_channels, 3, 1, 1)



    def forward(self, frame_coarse_feature, event_coarse_feature, last_fusion_feature):

        feature_f_e, feature_e_f = self.modality_weighting(frame_coarse_feature, event_coarse_feature)

        # concat_feature = torch.cat([feature_f_e, feature_e_f], dim=1)   # [B, C+C, H, W]
        feature_f_e_CA = self.event_CA(feature_e_f) * feature_f_e + feature_f_e
        feature_e_f_CA = self.frame_CA(feature_f_e) * feature_e_f + feature_e_f

        feature_f_e_CA_SA = self.event_SA(feature_e_f_CA) * feature_f_e_CA + feature_f_e_CA
        feature_e_f_CA_SA = self.frame_SA(feature_f_e_CA) * feature_e_f_CA + feature_e_f_CA

        concat_feature = torch.cat([feature_f_e_CA_SA, feature_e_f_CA_SA], dim=1)  # [B, C+C, H, W]


        if self.last_feature_channels is not None and last_fusion_feature is not None:
            concat_feature = torch.cat([concat_feature, torch.max_pool2d(last_fusion_feature, kernel_size=2)], dim=1)

        fused_feature = self.output_BCR(concat_feature)              # [B, output_channels, H, W]

        return fused_feature






class FusionEncoder_d8(nn.Module):
    def __init__(self, frame_channel_list, event_channel_list):
        super().__init__()
        ## frame_channels_list: [1, 16, 32, 64]
        ## shared_channel_list: [10, 9*8, 9*16, 9*32]
        assert len(frame_channel_list) == 4 and len(event_channel_list) == 4

        # self.output_channels = frame_channel_list[-1] * 2   # 128
        self.output_channels = frame_channel_list[-1]   # 64

        self.frame_encoder = Encoder_d8(frame_channel_list)
        self.event_encoder = Encoder_d8(event_channel_list)


        self.fusion_block_2 = FusionBlock(frame_channel_list[2], event_channel_list[2],
                                          output_channels=frame_channel_list[2],
                                          shared_channels=frame_channel_list[2],
                                          last_feature_channels=None)
        self.fusion_block_3 = FusionBlock(frame_channel_list[3], event_channel_list[3],
                                          output_channels=frame_channel_list[3],
                                          shared_channels=frame_channel_list[3],
                                          last_feature_channels=frame_channel_list[2])


    def forward(self, frame, event):
        _, frame_feature_2, frame_feature_3 = self.frame_encoder(frame)
        _, event_feature_2, event_feature_3 = self.event_encoder(event)

        fusion_feature_2 = self.fusion_block_2(frame_feature_2, event_feature_2, None)
        fusion_feature_3 = self.fusion_block_3(frame_feature_3, event_feature_3, fusion_feature_2)


        output_dict = {
            'frame_feature_list': [frame_feature_2, frame_feature_3],
            'event_feature_list': [event_feature_2, event_feature_3],
            'fusion_feature_list': [fusion_feature_2, fusion_feature_3]
        }

        return output_dict


    def reset(self, mask):
        self.fusion_block_2.reset(mask)
        self.fusion_block_3.reset(mask)

    def detach(self):
        self.fusion_block_2.detach()
        self.fusion_block_3.detach()




class Recurrent_layer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.recurrent_module = ConvRNN(input_channels, output_channels, cell='lstm')

    def forward(self, x):
        x = self.recurrent_module(x)
        return x

    def reset(self, mask):
        self.recurrent_module.reset(mask)

    def detach(self):
        self.recurrent_module.detach()



class MotionAwareSpatialChannelAttension(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MotionAwareSpatialChannelAttension, self).__init__()

        self.input_layer = Conv_BN_ReLU(input_channels, input_channels,
                                   kernel_size=3, stride=1, padding=1,
                                   bias=True)

        ## For spatial_pool
        self.conv_mask = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

        ## For attention_weight
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.LayerNorm([input_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, input_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )

        ## For output channels adjust
        self.output_layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )


    def spatial_pool(self, depth_feature):
        batch, channel, height, width = depth_feature.shape
        input_x = depth_feature                         # [N, C, H, W]
        input_x = input_x.view(batch, channel, height * width)  # [N, C, H * W]
        input_x = input_x.unsqueeze(1)                  # [N, 1, C, H * W]
        context_mask = self.conv_mask(depth_feature)    # [N, 1, H, W]
        context_mask = context_mask.view(batch, 1, height * width)  # [N, 1, H * W]
        context_mask = self.softmax(context_mask)       # [N, 1, H * W]
        context_mask = context_mask.unsqueeze(3)        # [N, 1, H * W, 1]
        # [N, 1, C, 1]
        # context attention
        context = torch.matmul(input_x, context_mask)   # [N, 1, C, H*W] * [N, 1, H*W, 1] -> # [N, 1, C, 1]
        context = context.view(batch, channel, 1, 1)    # [N, C, 1, 1]
        return context

    def forward(self, x):
        x = self.input_layer(x)

        attention_weight = self.spatial_pool(x)
        attention_weight = torch.sigmoid(self.channel_mul_conv(attention_weight))

        motion_feature = x * attention_weight
        motion_feature = self.output_layer(motion_feature)
        return motion_feature






class FeatureWarper(nn.Module):
    def __init__(self, feature_channels, event_feature_channels):
        super().__init__()
        deform_kernel_size = 3
        deform_groups = 8
        assert feature_channels % deform_groups == 0

        self.max_offset = 24   # Limit the range of offset

        ## 1. feature warping
        self.motion_extractor = MotionAwareSpatialChannelAttension(event_feature_channels, event_feature_channels)

        self.deformable_conv_offset_predictor = nn.Sequential(
            nn.Conv2d(event_feature_channels,
                      (deform_groups*deform_kernel_size*deform_kernel_size*2)*2,
                      kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d((deform_groups*deform_kernel_size*deform_kernel_size*2)*2,
                      deform_groups*deform_kernel_size*deform_kernel_size*2,
                      kernel_size=3, stride=1, padding=1),
        )

        self.deformable_conv = DeformConv2d(feature_channels, feature_channels,
                                            kernel_size=deform_kernel_size, stride=1, padding=deform_kernel_size//2,
                                            deform_groups=deform_groups)


    def forward(self, feature, event_feature):
        motion_inform = self.motion_extractor(event_feature)
        offsets = self.deformable_conv_offset_predictor(motion_inform)
        offsets = (F.sigmoid(offsets) * 2 - 1) * self.max_offset    # -max_offset ~ max_offset

        feature_warpped = self.deformable_conv(feature, offsets)

        return feature_warpped




class FrameEventNet(nn.Module):
    def __init__(self, frame_cin, exposure_event_cin, warping_event_cin, cout):
        super().__init__()
        assert exposure_event_cin % 2 == 0   # EST, pos and neg

        assert warping_event_cin % 2 == 0  # EST, pos and neg

        ## Some information
        self.desc_channels = 128
        self.scale_factor = 8
        self.cout = cout


        frame_channel_list = [frame_cin, 32, 64, 128]
        exposure_event_channel_list = [exposure_event_cin, 32, 64, 128]
        self.fusion_encoder = FusionEncoder_d8(frame_channel_list, exposure_event_channel_list)


        warpping_event_channel_list = [warping_event_cin, 32, 64, 128]
        self.warpping_event_encoder = Encoder_d8(warpping_event_channel_list)
        self.feature_warper = FeatureWarper(self.fusion_encoder.output_channels,
                                                 warpping_event_channel_list[-1])


        self.modality_weighting = ModalityWeightingModule(self.fusion_encoder.output_channels, warpping_event_channel_list[-1],
                                                          self.fusion_encoder.output_channels, warpping_event_channel_list[-1])
        self.second_fusion_CBR = Conv_BN_ReLU(self.fusion_encoder.output_channels+warpping_event_channel_list[-1],
                                              self.fusion_encoder.output_channels,
                                              kernel_size=3, stride=1, padding=1,
                                              bias=True)


        self.recurrent_layer = Recurrent_layer(self.fusion_encoder.output_channels,
                                               self.fusion_encoder.output_channels)

        self.final_conv = nn.Conv2d(self.fusion_encoder.output_channels, self.desc_channels,
                                    kernel_size=3, stride=1, padding=1)


        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(self.desc_channels, self.desc_channels,
                               kernel_size=4, stride=2, padding=1),
            Conv_BN_ReLU(self.desc_channels, 32,
                         kernel_size=3, stride=1, padding=1,
                         bias=True),

            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            Conv_BN_ReLU(32, 16,
                         kernel_size=3, stride=1, padding=1,
                         bias=True),

            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),

            nn.Sigmoid()
        )



    def forward(self, frame, exposure_event, warping_event_list):
        '''
        input:
            frame: [B, 1, H, W]
            exposure_event: [B, C1, H, W]
            warping_event_list: listlen:cout, dsgape[B, C2, H, W]
        '''
        assert len(warping_event_list) == self.cout
        fusion_feature_list = self.fusion_encoder(frame, exposure_event)['fusion_feature_list']
        fusion_feature = fusion_feature_list[-1]

        heatmap_list = []
        desc_list = []
        fusion_feature_warped_list = []
        for i in range(self.cout):    # forward
            sub_warping_events = warping_event_list[i]

            _, _, sub_warpping_event_feature = self.warpping_event_encoder(sub_warping_events)

            fusion_feature_warped = self.feature_warper(fusion_feature, sub_warpping_event_feature)


            fusion_feature_warped_weighted, sub_warpping_event_feature_weighted = self.modality_weighting(fusion_feature_warped, sub_warpping_event_feature)

            second_fusion_feature = self.second_fusion_CBR(torch.cat([fusion_feature_warped_weighted,
                                                                       sub_warpping_event_feature_weighted], dim=1))

            feature_i = second_fusion_feature + self.recurrent_layer(second_fusion_feature)
            feature_i = self.final_conv(feature_i)


            heatmap_i = self.heatmap_head(feature_i)
            heatmap_list.append(heatmap_i)

            desc_list.append(feature_i)

            fusion_feature_warped_list.append(fusion_feature_warped)


        output_dict = {
            'heatmaps': torch.cat(heatmap_list, dim=1),
            'desc_list': desc_list,
            'fusion_feature_list': fusion_feature_list,
            'fusion_feature_warped_list': fusion_feature_warped_list
        }
        return output_dict



    def reset(self, mask):
        self.recurrent_layer.reset(mask)

    def detach(self):
        self.recurrent_layer.detach()

