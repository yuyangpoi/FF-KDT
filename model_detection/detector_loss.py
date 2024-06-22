import pylab as pl
import torch
import torch.nn.functional as F
from model_detection.warp_utils import get_image_transform_tensor, scale_homography, warp_perspective_tensor  # , warp_points
from kornia.enhance import normalize_min_max
from kornia.geometry.transform import warp_grid
import math



## Use multiple N of different sizes to monitor consistency while conducting bidirectional supervision
def consistensy_loss_multi_scale_N(pred_heatmaps_batch,
                     rotation_vector, translation_vectors,
                     camera_nts, camera_depths,
                     camera_Ks, camera_Kinvs,
                     origin_sizes,
                     target_size,
                     N_list,
                     mask=None,
                     pred_features=None,
                     interval_list=None
                     ):
    '''
    :param pred_heatmaps_batch: [T=5, B, C=10, H, W]
    :param pose_batch: [T, B, C=10, ...]
    :param target_size: tuple(height, width)
    :param N: int
    :param mask:
    :param pred_features: [T=5*10, B, C=32, H, W] or None
    :return:
    '''
    def extract_patches(sal, patchsize):
        unfold = torch.nn.Unfold(patchsize, padding=0, stride=patchsize // 2)
        patches = unfold(sal).transpose(1, 2)  # flatten
        patches = F.normalize(patches, p=2, dim=2)  # norm [B, num, N*N], num=(height//(N//2)-1)*(width//(N//2)-1)
        return patches
    T, B, C, H, W = pred_heatmaps_batch.shape   # T is BPTT_T, C is heatmap_T
    assert rotation_vector.shape[:3] == (T, B, C)
    assert target_size[0] == H and target_size[1] == W, 'target_size: {}, H: {}, W: {}'.format(target_size, H, W)
    print('T, B, C, H, W: ', T, B, C, H, W)
    print('N_list: ', N_list)
    for N in N_list:
        assert H % (N//2) == 0 and W % (N//2) == 0

    if interval_list is None:
        interval_list = [1]


    heatmap_loss = 0
    for interval in interval_list:

        heatmap_0 = pred_heatmaps_batch.permute(0, 2, 1, 3, 4).reshape(T*C, B, H, W)[:T*C-interval].reshape((T*C-interval)*B, H, W).unsqueeze(1)
        rotation_vector_0 = rotation_vector.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[:T*C-interval].reshape((T*C-interval)*B, 3)
        translation_vectors_0 = translation_vectors.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[:T*C-interval].reshape((T*C-interval)*B, 3)
        camera_nts_0 = camera_nts.float().permute(0, 2, 1, 3, 4).reshape(T*C, B, 1, 3)[:T*C-interval].reshape((T*C-interval)*B, 1, 3)
        camera_depths_0 = camera_depths.float().permute(0, 2, 1, 3).reshape(T*C, B, 1)[:T*C-interval].reshape((T*C-interval)*B, 1)
        camera_Ks_0 = camera_Ks.float().permute(0, 2, 1, 3, 4).reshape(T*C, B, 3, 3)[:T*C-interval].reshape((T*C-interval)*B, 3, 3)
        camera_Kinvs_0 = camera_Kinvs.float().permute(0, 2, 1, 3, 4).reshape(T*C, B, 3, 3)[:T*C-interval].reshape((T*C-interval)*B, 3, 3)
        origin_sizes_0 = origin_sizes.float().permute(0, 2, 1, 3).reshape(T*C, B, 2)[:T*C-interval].reshape((T*C-interval)*B, 2)

        heatmap_1 = pred_heatmaps_batch.permute(0, 2, 1, 3, 4).reshape(T*C, B, H, W)[interval:].reshape((T*C-interval)*B, H, W).unsqueeze(1)
        rotation_vector_1 = rotation_vector.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[interval:].reshape((T*C-interval)*B, 3)
        translation_vectors_1 = translation_vectors.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[interval:].reshape((T*C-interval)*B, 3)

        target_sizes = torch.tensor(target_size).unsqueeze(0).repeat((T*C-interval)*B, 1).float().to(origin_sizes_0.device)


        visible_mask_0_1 = torch.ones_like(heatmap_0)  # [(T*C-interval)*B, 1, H, W]
        visible_mask_1_0 = torch.ones_like(heatmap_1)  # [(T*C-interval)*B, 1, H, W]


        ## 1. forward warp
        homo_0_1 = get_image_transform_tensor(rotation_vector_0, translation_vectors_0,
                                              rotation_vector_1, translation_vectors_1,
                                              camera_nts_0, camera_depths_0,
                                              camera_Ks_0, camera_Kinvs_0)
        homo_0_1_scaled = scale_homography(homo_0_1, origin_sizes_0, target_sizes)

        heatmap_0_warpped = warp_perspective_tensor(heatmap_0, homo_0_1_scaled, (H, W))
        visible_mask_0_1_warpped = warp_perspective_tensor(visible_mask_0_1, homo_0_1_scaled, (H, W))


        ## 2. backward warp
        homo_1_0 = get_image_transform_tensor(rotation_vector_1, translation_vectors_1,
                                              rotation_vector_0, translation_vectors_0,
                                              camera_nts_0, camera_depths_0,
                                              camera_Ks_0, camera_Kinvs_0)
        homo_1_0_scaled = scale_homography(homo_1_0, origin_sizes_0, target_sizes)

        heatmap_1_warpped = warp_perspective_tensor(heatmap_1, homo_1_0_scaled, (H, W))
        visible_mask_1_0_warpped = warp_perspective_tensor(visible_mask_1_0, homo_1_0_scaled, (H, W))



        for N in N_list:
            ## forward loss
            heatmap_1_patches = extract_patches(heatmap_1*visible_mask_0_1_warpped, N)  # [(T*C-interval)*B, patch_num, patchsize*patchsize]
            heatmap_0_warpped_patches = extract_patches(heatmap_0_warpped*visible_mask_0_1_warpped, N)

            heatmap_cosim_0_1 = (heatmap_1_patches * heatmap_0_warpped_patches).sum(dim=2)  # [(T*C-interval)*B, patch_num]
            heatmap_loss_interval_N_0_1 = 1 - heatmap_cosim_0_1.mean()


            ## backward loss
            heatmap_0_patches = extract_patches(heatmap_0 * visible_mask_1_0_warpped, N)  # [(T*C-interval)*B, patch_num, patchsize*patchsize]
            heatmap_1_warpped_patches = extract_patches(heatmap_1_warpped * visible_mask_1_0_warpped, N)

            heatmap_cosim_1_0 = (heatmap_0_patches * heatmap_1_warpped_patches).sum(dim=2)  # [(T*C-interval)*B, patch_num]
            heatmap_loss_interval_N_1_0 = 1 - heatmap_cosim_1_0.mean()

            heatmap_loss_interval_N = (heatmap_loss_interval_N_0_1 + heatmap_loss_interval_N_1_0) / 2.0
            heatmap_loss += heatmap_loss_interval_N


        if pred_features is not None:
            raise NotImplementedError


    return {'heatmap_loss': heatmap_loss / (len(interval_list)*len(N_list))}



def peaky_loss(pred_heatmaps_batch, N, valid_mask=None, alpha=0.5):
    '''

    :param pred_heatmaps_batch: [T, B, C, H, W]
    :param N: int
    :param valid_mask: [T, B, 1, H, W]
    :param alpha
    :return:
    '''

    assert N % 2 == 0, 'N must be even!'
    T, B, C, H, W = pred_heatmaps_batch.shape
    assert alpha > 0 and alpha <= 1
    # assert alpha == 1.0     # TODO: test

    heatmap = pred_heatmaps_batch.reshape(T*B, C, H, W)     # [T*B, C, H, W]
    processed_heatmap = F.avg_pool2d(heatmap, kernel_size=3, stride=1, padding=1)   # [T*B, C, H, W]
    max_heatmap = F.max_pool2d(processed_heatmap, kernel_size=N + 1, stride=1, padding=N // 2)
    mean_heatmap = F.avg_pool2d(processed_heatmap, kernel_size=N + 1, stride=1, padding=N // 2)

    if valid_mask is None:
        loss = 1 - (max_heatmap - mean_heatmap).mean()
    else:
        valid_mask = valid_mask.reshape(T*B, 1, H, W).float()   # [T*B, 1, H, W]
        valid_mask = F.max_pool2d(valid_mask, kernel_size=3, stride=1, padding=1)

        ## 1. positive area loss
        pos_loss = 1 - (max_heatmap - mean_heatmap) # [T*B, C, H, W]
        pos_loss = torch.divide(torch.sum(pos_loss * valid_mask, dim=(1, 2, 3)),
                                torch.sum(valid_mask + 1e-6, dim=(1, 2, 3)))
        pos_loss = torch.mean(pos_loss)

        ## 2. negative area loss
        if alpha < 1:
            neg_mask = 1 - valid_mask  # [T*B, 1, H, W]
            neg_loss = mean_heatmap
            neg_loss = torch.divide(torch.sum(neg_loss * neg_mask, dim=(1, 2, 3)),
                                    torch.sum(neg_mask + 1e-6, dim=(1, 2, 3)))
            neg_loss = torch.mean(neg_loss)

            print('pos_loss: ', pos_loss)
            print('neg_loss: ', neg_loss)

            loss = alpha * pos_loss + (1-alpha) * neg_loss
        else:
            loss = pos_loss
            print('just pos_loss: ', pos_loss)


    return loss




def consistency_loss_v3(pred_heatmaps_batch,
                       rotation_vector, translation_vectors,
                       camera_nts, camera_depths,
                       camera_Ks, camera_Kinvs,
                       origin_sizes,
                       target_size,
                       interval_list,
                       loss_class,
                       input_mask=None
                       ):
    T, B, C, H, W = pred_heatmaps_batch.shape  # T is BPTT_T, C is heatmap_T
    assert rotation_vector.shape[:3] == (T, B, C)
    assert target_size[0] == H and target_size[1] == W, 'target_size: {}, H: {}, W: {}'.format(target_size, H, W)
    print('T, B, C, H, W: ', T, B, C, H, W)
    print('loss_class.MSIP_sizes: ', loss_class.MSIP_sizes)
    print('loss_class.MSIP_factor_loss: ', loss_class.MSIP_factor_loss)


    loss_sum = 0
    for interval in interval_list:
        heatmap_0 = pred_heatmaps_batch.permute(0, 2, 1, 3, 4).reshape(T * C, B, H, W)[:T * C - interval].reshape((T * C - interval) * B, H, W).unsqueeze(1)
        rotation_vector_0 = rotation_vector.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[:T*C-interval].reshape((T*C-interval)*B, 3)
        translation_vectors_0 = translation_vectors.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[:T*C-interval].reshape((T*C-interval)*B, 3)
        camera_nts_0 = camera_nts.float().permute(0, 2, 1, 3, 4).reshape(T*C, B, 1, 3)[:T*C-interval].reshape((T*C-interval)*B, 1, 3)
        camera_depths_0 = camera_depths.float().permute(0, 2, 1, 3).reshape(T*C, B, 1)[:T*C-interval].reshape((T*C-interval)*B, 1)
        camera_Ks_0 = camera_Ks.float().permute(0, 2, 1, 3, 4).reshape(T*C, B, 3, 3)[:T*C-interval].reshape((T*C-interval)*B, 3, 3)
        camera_Kinvs_0 = camera_Kinvs.float().permute(0, 2, 1, 3, 4).reshape(T*C, B, 3, 3)[:T*C-interval].reshape((T*C-interval)*B, 3, 3)
        origin_sizes_0 = origin_sizes.float().permute(0, 2, 1, 3).reshape(T*C, B, 2)[:T*C-interval].reshape((T*C-interval)*B, 2)


        heatmap_1 = pred_heatmaps_batch.permute(0, 2, 1, 3, 4).reshape(T*C, B, H, W)[interval:].reshape((T*C-interval)*B, H, W).unsqueeze(1)
        rotation_vector_1 = rotation_vector.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[interval:].reshape((T*C-interval)*B, 3)
        translation_vectors_1 = translation_vectors.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[interval:].reshape((T*C-interval)*B, 3)


        target_sizes = torch.tensor(target_size).unsqueeze(0).repeat((T*C-interval)*B, 1).float().to(origin_sizes_0.device)

        ## Get src_2_dst and dst_2_src homography
        homo_0_1 = get_image_transform_tensor(rotation_vector_0, translation_vectors_0,
                                              rotation_vector_1, translation_vectors_1,
                                              camera_nts_0, camera_depths_0,
                                              camera_Ks_0, camera_Kinvs_0)
        homo_0_1_scaled = scale_homography(homo_0_1, origin_sizes_0, target_sizes)

        homo_1_0 = get_image_transform_tensor(rotation_vector_1, translation_vectors_1,
                                              rotation_vector_0, translation_vectors_0,
                                              camera_nts_0, camera_depths_0,
                                              camera_Ks_0, camera_Kinvs_0)
        homo_1_0_scaled = scale_homography(homo_1_0, origin_sizes_0, target_sizes)


        mask_borders = torch.ones_like(heatmap_0)
        loss_interval = loss_class(heatmap_0, heatmap_1, homo_0_1_scaled, homo_1_0_scaled, mask_borders)
        loss_sum += loss_interval


    return loss_sum / len(interval_list)




## This implementation is a warp coordinate grid
class Descriptor_loss(torch.nn.Module):
    def __init__(self, target_size, scale_factor, device):
        super().__init__()

        # self.interval_list = [1, 9]

        self.target_size = target_size
        self.scale_factor = scale_factor
        self.positive_margin = 1.0
        self.negative_margin = 0.2
        self.pos_lambda = 0.05

        H, W = self.target_size
        assert H % self.scale_factor == 0 and W % self.scale_factor == 0
        Hc, Wc = H // self.scale_factor, W // self.scale_factor

        coord_cells = torch.stack(torch.meshgrid([torch.arange(Wc, device=device),
                                                  torch.arange(Hc, device=device)], indexing='xy'), dim=-1)  # [Hc, Wc, 2(xy)]

        self.coord_cells = (coord_cells * scale_factor).unsqueeze(0)    # (B=1, Hc, Wc, 2(xy))
        print('self.coord_cells.shape: ', self.coord_cells.shape)
        print('self.coord_cells[:, 0, -1]: ', self.coord_cells[:, 0, -1])
        print('self.coord_cells[:, -1, 0]: ', self.coord_cells[:, -1, 0])
        print('self.coord_cells[:, -1, -1]: ', self.coord_cells[:, -1, -1])


    def forward(self,
                pred_features,
                rotation_vector, translation_vectors,
                camera_nts, camera_depths,
                camera_Ks, camera_Kinvs,
                origin_sizes,
                interval_list):
        '''
        :param pred_features: [T=5*10, B, C=32, H, W]
        :param pose_batch: [T, B, C=10, ...]
        :param
        :return:
        '''
        T, B, C = rotation_vector.shape[:3]   # T is BPTT_T, C is heatmap_T
        feature_C, feature_H, feature_W = pred_features.shape[-3:]
        device = pred_features.device
        assert pred_features.shape[0] == T*C
        assert self.target_size[0] // self.scale_factor == feature_H
        assert self.target_size[1] // self.scale_factor == feature_W


        loss = 0
        for interval in interval_list:

            pred_features_0 = pred_features[:T * C - interval].reshape((T * C - interval) * B, feature_C, feature_H, feature_W)
            rotation_vector_0 = rotation_vector.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[:T*C-interval].reshape((T*C-interval)*B, 3)
            translation_vectors_0 = translation_vectors.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[:T*C-interval].reshape((T*C-interval)*B, 3)
            camera_nts_0 = camera_nts.float().permute(0, 2, 1, 3, 4).reshape(T*C, B, 1, 3)[:T*C-interval].reshape((T*C-interval)*B, 1, 3)
            camera_depths_0 = camera_depths.float().permute(0, 2, 1, 3).reshape(T*C, B, 1)[:T*C-interval].reshape((T*C-interval)*B, 1)
            camera_Ks_0 = camera_Ks.float().permute(0, 2, 1, 3, 4).reshape(T*C, B, 3, 3)[:T*C-interval].reshape((T*C-interval)*B, 3, 3)
            camera_Kinvs_0 = camera_Kinvs.float().permute(0, 2, 1, 3, 4).reshape(T*C, B, 3, 3)[:T*C-interval].reshape((T*C-interval)*B, 3, 3)
            origin_sizes_0 = origin_sizes.float().permute(0, 2, 1, 3).reshape(T*C, B, 2)[:T*C-interval].reshape((T*C-interval)*B, 2)


            pred_features_1 = pred_features[interval:].reshape((T * C - interval) * B, feature_C, feature_H, feature_W)   # [(T*C-interval)*B, feature_C, feature_H, feature_W]
            rotation_vector_1 = rotation_vector.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[interval:].reshape((T*C-interval)*B, 3)
            translation_vectors_1 = translation_vectors.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[interval:].reshape((T*C-interval)*B, 3)

            target_sizes = torch.tensor(self.target_size).unsqueeze(0).repeat((T*C-interval)*B, 1).float().to(origin_sizes_0.device)


            homo_0_1 = get_image_transform_tensor(rotation_vector_0, translation_vectors_0,
                                                  rotation_vector_1, translation_vectors_1,
                                                  camera_nts_0, camera_depths_0,
                                                  camera_Ks_0, camera_Kinvs_0)
            ## Note that this is the size of the heatmap scale, as it warps the coordinate grid of the heatmap scale
            homo_0_1_scaled = scale_homography(homo_0_1, origin_sizes_0, target_sizes)  # [(T*C-interval)*B, 3, 3]


            pred_features_0 = pred_features_0.reshape((T * C - interval), B, feature_C, feature_H, feature_W)
            pred_features_1 = pred_features_1.reshape((T * C - interval), B, feature_C, feature_H, feature_W)
            homo_0_1_scaled = homo_0_1_scaled.reshape((T * C - interval), B, 3, 3)


            Hc, Wc = feature_H, feature_W   # H // self.scale_factor, W // self.scale_factor

            loss_a_interval = 0
            for tt in range((T*C-interval)):
                descriptor_0 = pred_features_0[tt]
                descriptor_1 = pred_features_1[tt]

                descriptor_0 = F.normalize(descriptor_0, p=2, dim=1)
                descriptor_1 = F.normalize(descriptor_1, p=2, dim=1)


                descriptor_0 = descriptor_0.unsqueeze(-1).unsqueeze(-1)     # [B, feature_C, Hc, Wc, 1, 1]
                descriptor_1 = descriptor_1.unsqueeze(-3).unsqueeze(-3)     # [B, feature_C, 1, 1, Hc, Wc]

                dot_product_desc = torch.sum(descriptor_0 * descriptor_1, dim=1)    # [B, C=1, Hc, Wc, Hc, Wc]
                dot_product_desc = F.relu(dot_product_desc.squeeze(1))              # [B, Hc, Wc, Hc, Wc]



                dot_product_desc = torch.reshape(F.normalize(torch.reshape(dot_product_desc, [B, Hc, Wc, Hc*Wc]), p=2, dim=3),
                                                 [B, Hc, Wc, Hc, Wc])
                dot_product_desc = torch.reshape(F.normalize(torch.reshape(dot_product_desc, [B, Hc*Wc, Hc, Wc]), p=2, dim=1),
                                                 [B, Hc, Wc, Hc, Wc])

                # print('dot_product_desc: \n', dot_product_desc)
                # print('torch.max(dot_product_desc): ', torch.max(dot_product_desc))
                # print('torch.mean(dot_product_desc): ', torch.mean(dot_product_desc))


                positive_dist = torch.maximum(torch.tensor(0., device=device), self.positive_margin - dot_product_desc)  # [1, Hc, Wc, Hc, Wc]
                negative_dist = torch.maximum(torch.tensor(0., device=device), dot_product_desc - self.negative_margin)


                ## gen gt_mask
                homo_0_1_tt = homo_0_1_scaled[tt]   # [B, 3, 3]
                coord_cells_tt = self.coord_cells.repeat(B, 1, 1, 1)   # [B, Hc, Wc, 2]

                warpped_coord_cells_tt = warp_grid(coord_cells_tt, homo_0_1_tt) # [1, Hc, Wc, 2]

                ## TODO:
                coord_cells_tt = coord_cells_tt.reshape(B, 1, 1, Hc, Wc, 2).float()                 # corresponding to descriptor_1
                warped_coord_cells_tt = warpped_coord_cells_tt.reshape(B, Hc, Wc, 1, 1, 2).float()  # corresponding to descriptor_0

                cell_distances = torch.norm(coord_cells_tt - warped_coord_cells_tt, dim=-1, p=2)    # [1, Hc, Wc, Hc, Wc]

                gt_mask = (cell_distances <= (self.scale_factor-0.5)).float()  #
                # print('interval: {}, torch.sum(gt_mask): {}'.format(interval, torch.sum(gt_mask)))


                pos_loss = gt_mask * positive_dist
                neg_loss = (1 - gt_mask) * negative_dist


                loss_a_interval_tt = self.pos_lambda * pos_loss + neg_loss

                loss_a_interval += torch.mean(loss_a_interval_tt)


            loss += (loss_a_interval / (T*C-interval))

        return loss / len(interval_list)




def feature_alignment_loss(feature1_list, feature2_list):
    '''

    Args:
        feature1_list: list dshape: [T, B, C, Hc, Wc]
        feature2_list: list, dshape: [T, B, C, Hc, Wc]

    Returns:
        feature_alignment_loss

    '''
    feature_num = len(feature1_list)
    assert feature_num == len(feature2_list)

    loss_func = torch.nn.MSELoss()

    loss_sum = 0
    for i in range(feature_num):
        loss_sum += loss_func(feature1_list[i], feature2_list[i])

    return loss_sum / feature_num







## TODO: test
def fusion_consistensy_loss_multi_scale_N(
            pred_heatmaps_batch,
        pred_feature_heatmaps_batch,
         rotation_vector, translation_vectors,
         camera_nts, camera_depths,
         camera_Ks, camera_Kinvs,
         origin_sizes,
         target_size,
         N_list,
         mask=None,
         pred_features=None,
         interval_list=None
                     ):
    '''
    :param pred_heatmaps_batch: [T=5, B, C=10, H, W]
    :param pose_batch: [T, B, C=10, ...]
    :param target_size: tuple(height, width)
    :param N: int
    :param mask:
    :param pred_features: [T=5*10, B, C=32, H, W] or None
    :return:
    '''
    def extract_patches(sal, patchsize):
        unfold = torch.nn.Unfold(patchsize, padding=0, stride=patchsize // 2)
        patches = unfold(sal).transpose(1, 2)  # flatten
        patches = F.normalize(patches, p=2, dim=2)  # norm [B, num, N*N], num=(height//(N//2)-1)*(width//(N//2)-1)
        return patches
    T, B, C, H, W = pred_heatmaps_batch.shape   # T is BPTT_T, C is heatmap_T
    assert pred_feature_heatmaps_batch.shape == pred_heatmaps_batch.shape

    assert rotation_vector.shape[:3] == (T, B, C)
    assert target_size[0] == H and target_size[1] == W, 'target_size: {}, H: {}, W: {}'.format(target_size, H, W)
    print('T, B, C, H, W: ', T, B, C, H, W)
    print('N_list: ', N_list)
    for N in N_list:
        assert H % (N//2) == 0 and W % (N//2) == 0

    if interval_list is None:
        interval_list = [1]


    heatmap_loss = 0
    feature_heatmap_loss = 0
    for interval in interval_list:

        heatmap_0 = pred_heatmaps_batch.permute(0, 2, 1, 3, 4).reshape(T*C, B, H, W)[:T*C-interval].reshape((T*C-interval)*B, H, W).unsqueeze(1)
        rotation_vector_0 = rotation_vector.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[:T*C-interval].reshape((T*C-interval)*B, 3)
        translation_vectors_0 = translation_vectors.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[:T*C-interval].reshape((T*C-interval)*B, 3)
        camera_nts_0 = camera_nts.float().permute(0, 2, 1, 3, 4).reshape(T*C, B, 1, 3)[:T*C-interval].reshape((T*C-interval)*B, 1, 3)
        camera_depths_0 = camera_depths.float().permute(0, 2, 1, 3).reshape(T*C, B, 1)[:T*C-interval].reshape((T*C-interval)*B, 1)
        camera_Ks_0 = camera_Ks.float().permute(0, 2, 1, 3, 4).reshape(T*C, B, 3, 3)[:T*C-interval].reshape((T*C-interval)*B, 3, 3)
        camera_Kinvs_0 = camera_Kinvs.float().permute(0, 2, 1, 3, 4).reshape(T*C, B, 3, 3)[:T*C-interval].reshape((T*C-interval)*B, 3, 3)
        origin_sizes_0 = origin_sizes.float().permute(0, 2, 1, 3).reshape(T*C, B, 2)[:T*C-interval].reshape((T*C-interval)*B, 2)

        heatmap_1 = pred_heatmaps_batch.permute(0, 2, 1, 3, 4).reshape(T*C, B, H, W)[interval:].reshape((T*C-interval)*B, H, W).unsqueeze(1)
        rotation_vector_1 = rotation_vector.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[interval:].reshape((T*C-interval)*B, 3)
        translation_vectors_1 = translation_vectors.float().permute(0, 2, 1, 3).reshape(T*C, B, 3)[interval:].reshape((T*C-interval)*B, 3)

        target_sizes = torch.tensor(target_size).unsqueeze(0).repeat((T*C-interval)*B, 1).float().to(origin_sizes_0.device)


        visible_mask_0_1 = torch.ones_like(heatmap_0)  # [(T*C-interval)*B, 1, H, W]
        visible_mask_1_0 = torch.ones_like(heatmap_1)  # [(T*C-interval)*B, 1, H, W]


        ##
        feature_heatmap_0 = pred_feature_heatmaps_batch.permute(0, 2, 1, 3, 4).reshape(T*C, B, H, W)[:T*C-interval].reshape(
            (T*C-interval)*B, H, W).unsqueeze(1)
        feature_heatmap_1 = pred_feature_heatmaps_batch.permute(0, 2, 1, 3, 4).reshape(T * C, B, H, W)[interval:].reshape(
            (T * C - interval) * B, H, W).unsqueeze(1)


        homo_0_1 = get_image_transform_tensor(rotation_vector_0, translation_vectors_0,
                                              rotation_vector_1, translation_vectors_1,
                                              camera_nts_0, camera_depths_0,
                                              camera_Ks_0, camera_Kinvs_0)
        homo_0_1_scaled = scale_homography(homo_0_1, origin_sizes_0, target_sizes)

        heatmap_0_warpped = warp_perspective_tensor(heatmap_0, homo_0_1_scaled, (H, W))
        visible_mask_0_1_warpped = warp_perspective_tensor(visible_mask_0_1, homo_0_1_scaled, (H, W))

        feature_heatmap_0_warpped = warp_perspective_tensor(feature_heatmap_0, homo_0_1_scaled, (H, W))


        homo_1_0 = get_image_transform_tensor(rotation_vector_1, translation_vectors_1,
                                              rotation_vector_0, translation_vectors_0,
                                              camera_nts_0, camera_depths_0,
                                              camera_Ks_0, camera_Kinvs_0)
        homo_1_0_scaled = scale_homography(homo_1_0, origin_sizes_0, target_sizes)

        heatmap_1_warpped = warp_perspective_tensor(heatmap_1, homo_1_0_scaled, (H, W))
        visible_mask_1_0_warpped = warp_perspective_tensor(visible_mask_1_0, homo_1_0_scaled, (H, W))

        feature_heatmap_1_warpped = warp_perspective_tensor(feature_heatmap_1, homo_1_0_scaled, (H, W))



        for N in N_list:
            ## 1. heatmap loss
            heatmap_1_patches = extract_patches(heatmap_1*visible_mask_0_1_warpped, N)  # [(T*C-interval)*B, patch_num, patchsize*patchsize]
            heatmap_0_warpped_patches = extract_patches(heatmap_0_warpped*visible_mask_0_1_warpped, N)

            heatmap_cosim_0_1 = (heatmap_1_patches * heatmap_0_warpped_patches).sum(dim=2)  # [(T*C-interval)*B, patch_num]
            heatmap_loss_interval_N_0_1 = 1 - heatmap_cosim_0_1.mean()

            heatmap_0_patches = extract_patches(heatmap_0 * visible_mask_1_0_warpped, N)  # [(T*C-interval)*B, patch_num, patchsize*patchsize]
            heatmap_1_warpped_patches = extract_patches(heatmap_1_warpped * visible_mask_1_0_warpped, N)

            heatmap_cosim_1_0 = (heatmap_0_patches * heatmap_1_warpped_patches).sum(dim=2)  # [(T*C-interval)*B, patch_num]
            heatmap_loss_interval_N_1_0 = 1 - heatmap_cosim_1_0.mean()


            heatmap_loss_interval_N = (heatmap_loss_interval_N_0_1 + heatmap_loss_interval_N_1_0) / 2.0
            heatmap_loss += heatmap_loss_interval_N


            ## 2. feature heatmap loss
            feature_heatmap_1_patches = extract_patches(feature_heatmap_1 * visible_mask_0_1_warpped, N)
            feature_heatmap_0_warpped_patches = extract_patches(feature_heatmap_0_warpped*visible_mask_0_1_warpped, N)
            feature_heatmap_cosim_0_1 = (feature_heatmap_1_patches * feature_heatmap_0_warpped_patches).sum(dim=2)
            feature_heatmap_loss_interval_N_0_1 = 1 - feature_heatmap_cosim_0_1.mean()

            feature_heatmap_0_patches = extract_patches(feature_heatmap_0 * visible_mask_1_0_warpped, N)
            feature_heatmap_1_warpped_patches = extract_patches(feature_heatmap_1_warpped * visible_mask_1_0_warpped, N)

            feature_heatmap_cosim_1_0 = (feature_heatmap_0_patches * feature_heatmap_1_warpped_patches).sum(dim=2)
            feature_heatmap_loss_interval_N_1_0 = 1 - feature_heatmap_cosim_1_0.mean()

            feature_heatmap_loss_interval_N = (feature_heatmap_loss_interval_N_0_1 + feature_heatmap_loss_interval_N_1_0) / 2.0
            feature_heatmap_loss += feature_heatmap_loss_interval_N




        if pred_features is not None:
            raise NotImplementedError


    return {
        'heatmap_loss': heatmap_loss / (len(interval_list)*len(N_list)),
        'feature_heatmap_loss': feature_heatmap_loss / (len(interval_list)*len(N_list))
    }







































