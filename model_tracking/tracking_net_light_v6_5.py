import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_tracking.tracking_utils import extract_query_vector, crop_feature_map, compute_correlation_map


class ResidualBlock(nn.Module):
    def __init__(self, net, norm, activation_func):
        super(ResidualBlock, self).__init__()
        self.net = net
        self.norm = norm
        self.activation_func = activation_func

    def forward(self, x):
        to_skip = x
        out = self.norm(self.net(x))
        out = out + to_skip
        out = self.activation_func(out)
        return out




class BaseNet_v3(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        block_num = 6
        assert block_num >= 1


        self.input_layer = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size=5, padding=2),
            nn.InstanceNorm1d(input_channels),
            nn.GELU()
        )

        self.conv_net = nn.Sequential()
        for i in range(block_num):
            self.conv_net.append(
                nn.Sequential(
                    ResidualBlock(
                        net=nn.Conv1d(input_channels, input_channels, kernel_size=1),
                        norm=nn.InstanceNorm1d(input_channels),
                        activation_func=nn.GELU()
                    ),
                    ResidualBlock(
                        net=nn.Sequential(
                            nn.Conv1d(input_channels, input_channels, kernel_size=5, padding=2, groups=input_channels),
                            nn.Conv1d(input_channels, input_channels, kernel_size=1),
                        ),
                        norm=nn.InstanceNorm1d(input_channels),
                        activation_func=nn.GELU()
                    )
                )
            )

        self.output_layer = nn.Conv1d(input_channels, output_channels, kernel_size=1, padding=0)


    def forward(self, x):
        x = self.input_layer(x)
        x = self.conv_net(x)
        x = self.output_layer(x)
        return x




class FrameEventTrackingNet(nn.Module):
    def __init__(self, feature_channels, scale_factor,
                 patch_size,
                 iterations,
                 pyramid_layers,
                 correlation_map_mann='dot_product'):
        super().__init__()

        self.feature_channels = feature_channels
        self.scale_factor = scale_factor

        self.patch_size = patch_size

        self.iterations = iterations

        self.embedding_channels = 64

        assert pyramid_layers >= 1
        self.pyramid_layers = pyramid_layers

        conv_input_channels = 2 + 2 + self.embedding_channels + feature_channels + patch_size * patch_size * self.pyramid_layers
        conv_output_chennels = 2 + feature_channels
        self.conv_net = BaseNet_v3(conv_input_channels, conv_output_chennels)

        self.correlation_map_mann = correlation_map_mann

        ## TODO:
        assert self.correlation_map_mann == 'dot_product'
        # assert self.correlation_map_mann == 'cosim'


    @staticmethod
    def extract_query_points(trajectory, query_idx):
        '''
        input:
            trajectory: [T, B, 2(xy), N]
            query_idx: [B, N], 0<= query_idx<=T-1 or None
        output:
            query_points: [B, 2(xy), N]
        '''
        query_idx_repeat = query_idx.unsqueeze(0).unsqueeze(2).repeat(1, 1, 2, 1)   # [1, B, 2, N]
        query_points = torch.gather(trajectory, 0, query_idx_repeat).squeeze(0)   # [1, B, 2, N]
        return query_points



    def base_forward(self, query_vectors_init, scaled_traj_init, features, T):
        '''
        input:
            query_vectors_init: [T*B*N, C]
            scaled_traj_init: [T*B*N, 2], already_scaled
            features: [T*B*N, C, Hc, Wc]
        output:
            scaled_traj_list: listlen: iter, dtype: [T*B*N, 2]
        '''
        TBN, C = query_vectors_init.shape
        BN = TBN // T

        query_vectors_i = query_vectors_init
        traj_i = scaled_traj_init  # [T*B*N, 2]


        ## 生成特征金字塔
        features_pyramid_list = [features]
        for _ in range(self.pyramid_layers - 1):
            features = F.avg_pool2d(features, kernel_size=2, stride=2)
            features_pyramid_list.append(features)


        scaled_traj_list = []
        for i in range(self.iterations):
            traj_i = traj_i.detach()    # TODO: test

            traj_i_t0 = traj_i.reshape(T, BN, 2)[0].unsqueeze(0)    # [1, B*N, 2]
            flow = traj_i.reshape(T, BN, 2) - traj_i_t0             # [T, B*N, 2]
            flow = flow.reshape(TBN, 2)                             # [T*B*N, 2]


            correlation_vectors_i_list = []  # len:P
            for p in range(self.pyramid_layers):
                features_cropped_i_p = crop_feature_map(features_pyramid_list[p],
                                                        traj_i / (2 ** p),
                                                        self.patch_size)  # [T*B*N, C, crop_patch_size, crop_patch_size]

                correlation_map_i_p = compute_correlation_map(query_vectors_i.unsqueeze(-1).unsqueeze(-1),
                                                              features_cropped_i_p,
                                                              mode=self.correlation_map_mann)  # [T*B*N, 1, patchsize, patchsize]

                # correlation_map_i_p = F.relu(correlation_map_i_p) # TODO: test
                correlation_vectors_i_p = correlation_map_i_p.reshape(TBN,
                                                                      self.patch_size * self.patch_size)  # [T*B*N, K=patchsize*patchsize]
                correlation_vectors_i_list.append(correlation_vectors_i_p)

            total_vector_i = torch.cat([traj_i,
                                        flow,
                                        self.embedding_sincos(flow, channels=self.embedding_channels),
                                        query_vectors_i] + correlation_vectors_i_list,
                                       dim=-1)  # [T*B*N, 2+2+emb_C+C+K=2+2+64+128+49*P]
            total_vector_i = total_vector_i.reshape(T, BN, -1).permute(1, 2, 0)  # [B*N, 2+C+K=2+256+49*P, T]
            res_feature = self.conv_net(total_vector_i)  # [B*N, 2+C, T]
            res_feature = res_feature.permute(2, 0, 1).reshape(TBN, 2 + C)  # [T*B*N, 2+C]

            res_traj = res_feature[:, :2]  # [T*B*N, 2]
            res_query_vectors = res_feature[:, 2:]  # [T*B*N, C]

            traj_i = traj_i + res_traj  # [T*B*N, 2]
            query_vectors_i = query_vectors_i + res_query_vectors  # [T*B*N, C]

            scaled_traj_list.append(traj_i)

        return scaled_traj_list


    def embedding_sincos(self, pos, channels, temperature=10000):
        '''
        input:
            pos: [B, 2]
        output:
            pos_emb: [B, 2]
        '''
        assert pos.shape[-1] == 2
        assert (channels % 4) == 0, 'channels{} must be multiple of 4 for sincos emb'.format(channels)

        x = pos[..., 0]
        y = pos[..., 1]

        omega = torch.arange(channels // 4, device=pos.device) / (channels // 4 - 1)    # [channels//4]
        omega = 1. / (temperature ** omega)

        x = x.unsqueeze(-1) * omega.unsqueeze(0)    # [B, channels//4]
        y = y.unsqueeze(-1) * omega.unsqueeze(0)    # [B, channels//4]

        pos_emb = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)    # [B, channels]
        return pos_emb



    def extract_query_vectors(self, query_idx, query_points, features):
        '''
        input:
            query_idx: [B, N]
            query_points: [B, 2, N]
            features: [T, B, C, H//scale_factor, W//scale_factor]
        output:
            query_vectors: [B, C, N]
        '''
        T, B, C, Hc, Wc = features.shape
        N = query_idx.shape[-1]
        assert query_points.shape[-1] == N

        query_idx_repeat = query_idx.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, C, Hc, Wc, 1)
        query_points_scaled = query_points / self.scale_factor  # [B, 2, N]

        features_repeat = features.unsqueeze(-1).repeat(1, 1, 1, 1, 1, N)   # [T, B, C, Hc, Wc, N]
        features_at_query_idx = torch.gather(features_repeat, 0, query_idx_repeat).squeeze(0)  # [B, C, Hc, Wc, N]
        features_at_query_idx = features_at_query_idx.permute(0, 4, 1, 2, 3).reshape(B * N, C, Hc, Wc)

        query_vectors = extract_query_vector(query_points_scaled.permute(0, 2, 1).reshape(B * N, 2),
                                             features_at_query_idx).squeeze(-1).squeeze(-1)  # [B*N, C]
        query_vectors = query_vectors.reshape(B, N, C).permute(0, 2, 1)     # [B, C, N]

        return query_vectors


    def forward(self, traj_init,
                query_vectors_init,
                features):
        '''
        input:
            traj_init: [T, B, 2, N]
            query_vectors_init: [B, C, N]
            features: [T, B, C, H//scale_factor, W//scale_factor]
        output:
            trajectory: listlen: iter, dtype: [T, B, 2, N]
        '''
        T, B, C, Hc, Wc = features.shape
        N = traj_init.shape[-1]

        assert traj_init.shape[0] == features.shape[0], \
            'traj_init.shape[0]: {}, features.shape[0]: {}'.format(traj_init.shape[0], features.shape[0])
        assert traj_init.shape[-1] == query_vectors_init.shape[-1], \
            'traj_init.shape[-1]: {}, features.shape[-1]: {}'.format(traj_init.shape[-1], features.shape[-1])


        ## 1. Repeat and Reshape traj_init, query_vectors_init, features
        features_repeat = features.unsqueeze(-1).repeat(1, 1, 1, 1, 1, N)  # [T, B, C, Hc, Wc, N]
        features_repeat_reshape = features_repeat.permute(0, 1, 5, 2, 3, 4).reshape(T * B * N, C, Hc, Wc)       # [T*B*N, C, Hc, Wc]

        traj_init_scaled = traj_init / self.scale_factor  # [T, B, 2, N]
        traj_init_scaled_repeat_reshape = traj_init_scaled.permute(0, 1, 3, 2).reshape(T * B * N, 2)            # [T*B*N, 2]

        query_vectors_init_repeat = query_vectors_init.unsqueeze(0).repeat(T, 1, 1, 1)          # [T, B, C, N]
        query_vectors_init_repeat_reshape = query_vectors_init_repeat.permute(0, 1, 3, 2).reshape(T * B * N, C)     # [T*B*N, C]


        ## 2. Adjust trajectory
        scaled_pred_traj_list = self.base_forward(query_vectors_init_repeat_reshape, traj_init_scaled_repeat_reshape, features_repeat_reshape, T)


        trajectory_each_iter_list = []
        for scaled_pred_traj in scaled_pred_traj_list:
            trajectory_each_iter_list.append(scaled_pred_traj.view(T, B, N, 2).permute(0, 1, 3, 2) * self.scale_factor) # [T, B, 2, N]


        return trajectory_each_iter_list

































