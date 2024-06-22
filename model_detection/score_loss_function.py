import math, torch
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective


def ones_multiple_channels(size, num_channels):

    ones = torch.ones((size, size))
    weights = torch.zeros((num_channels, num_channels, size, size), dtype=torch.float32)

    for i in range(num_channels):
        weights[i, i, :, :] = ones

    return weights

def grid_indexes(size):

    weights = torch.zeros((2, 1, size, size), dtype=torch.float32)

    columns = []
    for idx in range(1, 1+size):
        columns.append(torch.ones((size))*idx)
    columns = torch.stack(columns)

    rows = []
    for idx in range(1, 1+size):
        rows.append(torch.tensor(range(1, 1+size)))
    rows = torch.stack(rows)

    weights[0, 0, :, :] = columns
    weights[1, 0, :, :] = rows

    return weights


def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2

def linear_upsample_weights(half_factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with linear filter
    initialization.
    """
    filter_size = get_kernel_size(half_factor)

    weights = torch.zeros((number_of_classes,
                        number_of_classes,
                        filter_size,
                        filter_size,
                        ), dtype=torch.float32)

    upsample_kernel = torch.ones((filter_size, filter_size))
    for i in range(number_of_classes):
        weights[i, i, :, :] = upsample_kernel

    return weights


class Kernels_custom:
    def __init__(self, batchsize, MSIP_sizes=[]):

        self.batch_size = batchsize
        # create_kernels
        self.kernels = {}

        if MSIP_sizes != []:
            self.create_kernels(MSIP_sizes)

        if 8 not in MSIP_sizes:
            self.create_kernels([8])


    def create_kernels(self, MSIP_sizes):
        # Grid Indexes for MSIP
        for ksize in MSIP_sizes:

            ones_kernel = ones_multiple_channels(ksize, 1)
            indexes_kernel = grid_indexes(ksize)
            upsample_filter_np = linear_upsample_weights(int(ksize / 2), 1)

            self.ones_kernel = ones_kernel.requires_grad_(False)
            self.kernels['ones_kernel_'+str(ksize)] = self.ones_kernel

            self.upsample_filter_np = upsample_filter_np.requires_grad_(False)
            self.kernels['upsample_filter_np_'+str(ksize)] = self.upsample_filter_np

            self.indexes_kernel = indexes_kernel.requires_grad_(False)
            self.kernels['indexes_kernel_'+str(ksize)] = self.indexes_kernel


    def get_kernels(self, device):
        kernels = {}
        for k,v in self.kernels.items():
            kernels[k] = v.to(device)
        return kernels


class KeypointDetectionLoss:
    def __init__(self, batchsize, MSIP_sizes, MSIP_factor_loss, patch_size, device):
        # custom_kernels = Kernels_custom(args.batch_size, args.MSIP_sizes)
        # kernels = custom_kernels.get_kernels(device)  ## with GPU
        #
        # self.kernels = kernels
        #
        # self.MSIP_sizes = args.MSIP_sizes
        # self.MSIP_factor_loss = args.MSIP_factor_loss
        # self.patch_size = args.patch_size
        custom_kernels = Kernels_custom(batchsize, MSIP_sizes)
        kernels = custom_kernels.get_kernels(device)  ## with GPU

        self.kernels = kernels

        self.MSIP_sizes = MSIP_sizes
        self.MSIP_factor_loss = MSIP_factor_loss
        self.patch_size = patch_size

    def __call__(self, features_k1, features_k2, h_src_2_dst, h_dst_2_src, mask_borders):
        keynet_loss = 0
        for MSIP_idx, (MSIP_size, MSIP_factor) in enumerate(zip(self.MSIP_sizes, self.MSIP_factor_loss)):
            MSIP_loss = self.ip_loss(features_k1, features_k2, MSIP_size, h_src_2_dst, h_dst_2_src, mask_borders)

            keynet_loss += MSIP_factor * MSIP_loss

            # MSIP_level_name = "MSIP_ws_{}".format(MSIP_size)
            # print("MSIP_level_name {} of MSIP_idx {} : {}, {} ".format(MSIP_level_name, MSIP_idx,  MSIP_loss, MSIP_factor * MSIP_loss)) ## logging

        return keynet_loss

    def ip_loss(self, src_score_maps, dst_score_maps, window_size, h_src_2_dst, h_dst_2_src, mask_borders):
        src_maps, dst_maps, mask_borders = check_divisible(src_score_maps, dst_score_maps,
                                                           mask_borders,
                                                           self.patch_size, window_size)

        warped_output_shape = src_maps.shape[2:]    # [H, W]

        ## Note that warp_perspective function is not inverse warping! as different with tensorflow.image.transform
        src_maps_warped = warp_perspective(src_maps * mask_borders, h_src_2_dst, dsize=warped_output_shape)
        dst_maps_warped = warp_perspective(dst_maps * mask_borders, h_dst_2_src, dsize=warped_output_shape)
        visible_src_mask = warp_perspective(mask_borders, h_dst_2_src, dsize=warped_output_shape) * mask_borders
        visible_dst_mask = warp_perspective(mask_borders, h_src_2_dst, dsize=warped_output_shape) * mask_borders

        # Remove borders and stop gradients to only backpropagate on the unwarped maps
        src_maps = visible_src_mask * src_maps
        dst_maps = visible_dst_mask * dst_maps
        src_maps_warped = visible_dst_mask * src_maps_warped.detach()
        dst_maps_warped = visible_src_mask * dst_maps_warped.detach()

        # Use IP Layer to extract soft coordinates from original maps & Compute soft weights
        src_indexes, weights_src = self.ip_layer(src_maps, window_size)
        dst_indexes, weights_dst = self.ip_layer(dst_maps, window_size)

        # Use argmax layer to extract NMS coordinates from warped maps
        src_indexes_nms_warped = self.grid_indexes_nms_conv(src_maps_warped, window_size)
        dst_indexes_nms_warped = self.grid_indexes_nms_conv(dst_maps_warped, window_size)

        # Multiply weights with the visible coordinates to discard uncommon regions
        weights_src = min_max_norm(weights_src) * max_pool2d(visible_src_mask, window_size)[0]
        weights_dst = min_max_norm(weights_dst) * max_pool2d(visible_dst_mask, window_size)[0]

        loss_src = self.compute_loss(src_indexes, dst_indexes_nms_warped, weights_src, window_size)
        loss_dst = self.compute_loss(dst_indexes, src_indexes_nms_warped, weights_dst, window_size)

        # print('loss_src: ', loss_src)
        # print('loss_dst: ', loss_dst)
        loss_indexes = (loss_src + loss_dst) / 2.

        return loss_indexes

    # ## Obtain soft selected index  (Index Proposal Layer)
    # def ip_layer(self, scores_relu, window_size):
    #     weights, _ = max_pool2d(scores_relu, window_size)    # [B, 1, window_num, window_num], window_num=H/window_size
    #     max_pool_unpool = F.conv_transpose2d(weights, self.kernels['upsample_filter_np_' + str(window_size)],
    #                                          stride=[window_size, window_size]) # [B, 1, H, W], 0~1
    #
    #     scores_norm = torch.div(scores_relu, max_pool_unpool + 1e-6)     # 0~1
    #     exp_map_1 = torch.add(torch.pow(math.e, scores_norm), -1 * (1. - 1e-6))   # [B, 1, H, W]
    #
    #     sum_exp_map_1 = F.conv2d(exp_map_1, self.kernels['ones_kernel_' + str(window_size)],
    #                              stride=[window_size, window_size], padding=0)      # [B, 1, window_num, window_num]
    #
    #     indexes_map = F.conv2d(exp_map_1, self.kernels['indexes_kernel_' + str(window_size)],
    #                            stride=[window_size, window_size], padding=0)        # [B, 2, window_num, window_num], 1~window_size
    #
    #     indexes_map = torch.div(indexes_map, torch.add(sum_exp_map_1, 1e-6))
    #
    #     ## compute soft-score
    #     sum_scores_map_1 = F.conv2d(exp_map_1 * scores_relu, self.kernels['ones_kernel_' + str(window_size)],
    #                                 stride=[window_size, window_size], padding=0)
    #     soft_scores = torch.div(sum_scores_map_1, torch.add(sum_exp_map_1, 1e-6))   # [B, 1, window_num, window_num]
    #
    #     return indexes_map, soft_scores.detach()
    ## Obtain soft selected index  (Index Proposal Layer)
    def ip_layer(self, scores_sigmoid, window_size):
        '''
        Args:
            scores_sigmoid: The heatmap[B, 1, H, W] has been processed through sigmoid
            window_size: int
        Returns:
            indexes_map: [B, 2, window_num, window_num]
            soft_scores: [B, 1, window_num, window_num]
        '''
        # print('\nwindow_size: ', window_size)
        # weights, _ = max_pool2d(scores, window_size)    # [B, 1, window_num, window_num], window_num=H/window_size
        # max_pool_unpool = F.conv_transpose2d(weights, self.kernels['upsample_filter_np_' + str(window_size)],
        #                                      stride=[window_size, window_size])  # [B, 1, H, W], 0~1
        # scores_norm = torch.div(scores, max_pool_unpool + 1e-6)  # 0~1
        # exp_map_1 = torch.add(torch.pow(math.e, scores_norm), -1 * (1. - 1e-6))   # [B, 1, H, W], 1e-6~(math.e-1)

        softmax_temperature = 10.0
        exp_map_1 = torch.pow(math.e, softmax_temperature*scores_sigmoid)  # [B, 1, H, W]
        # print('-1 * (1. - 1e-6): ', -1 * (1. - 1e-6))  # 0.9999
        # print('torch.max(exp_map_1): ', torch.max(exp_map_1))
        # print('torch.min(exp_map_1): ', torch.min(exp_map_1))

        sum_exp_map_1 = F.conv2d(exp_map_1, self.kernels['ones_kernel_' + str(window_size)],
                                 stride=[window_size, window_size], padding=0)      # [B, 1, window_num, window_num]
        # print('torch.max(sum_exp_map_1): ', torch.max(sum_exp_map_1))
        # print('torch.min(sum_exp_map_1): ', torch.min(sum_exp_map_1))

        indexes_map = F.conv2d(exp_map_1, self.kernels['indexes_kernel_' + str(window_size)],
                               stride=[window_size, window_size], padding=0)        # [B, 2, window_num, window_num]

        indexes_map = torch.div(indexes_map, torch.add(sum_exp_map_1, 1e-6))        # [B, 2, window_num, window_num], 1~window_size

        ## compute soft-score
        sum_scores_map_1 = F.conv2d(exp_map_1 * scores_sigmoid, self.kernels['ones_kernel_' + str(window_size)],
                                    stride=[window_size, window_size], padding=0)

        soft_scores = torch.div(sum_scores_map_1, torch.add(sum_exp_map_1, 1e-6))   # [B, 1, window_num, window_num]

        return indexes_map, soft_scores.detach()




    # Obtain hard selcted index by argmax in window
    def grid_indexes_nms_conv(self, scores, window_size):
        weights, indexes = max_pool2d(scores, window_size)

        weights_norm = torch.div(weights, torch.add(weights, torch.finfo(float).eps))

        score_map = F.max_unpool2d(weights_norm, indexes, kernel_size=[window_size, window_size])

        indexes_label = F.conv2d(score_map, self.kernels['indexes_kernel_' + str(window_size)],
                                 stride=[window_size, window_size], padding=0)

        # ### To prevent too many the upper-left coordinates cases
        # ind_rand = torch.randint(low=0, high=window_size, size=indexes_label.shape, dtype=torch.int32).to(torch.float32).to(indexes_label.device)

        # indexes_label = torch.where((indexes_label == torch.zeros_like(indexes_label)), ind_rand, indexes_label)

        return indexes_label

    @staticmethod
    def compute_loss(src_indexes, label_indexes, weights_indexes, window_size):
        ## loss_ln_indexes_norm
        norm_sq = torch.pow((src_indexes - label_indexes) / window_size, 2)
        norm_sq = torch.sum(norm_sq, dim=1, keepdims=True)
        weigthed_norm_sq = 1000 * (torch.multiply(weights_indexes, norm_sq))
        loss = torch.mean(weigthed_norm_sq)

        return loss


def max_pool2d(scores, window_size):
    ## stride is same as kernel_size as default.
    weights, indexes = F.max_pool2d(scores, kernel_size=(window_size, window_size), padding=0, return_indices=True)

    return weights, indexes


def check_divisible(src_maps, dst_maps, mask_borders, patch_size, window_size):
    # Check if patch size is divisible by the window size
    if patch_size % window_size > 0:
        batch_shape = src_maps.shape
        new_size = patch_size - (patch_size % window_size)
        src_maps = src_maps[0:batch_shape[0], 0:batch_shape[1], 0:new_size, 0:new_size]
        dst_maps = dst_maps[0:batch_shape[0], 0:batch_shape[1], 0:new_size, 0:new_size]
        mask_borders = mask_borders[0:batch_shape[0], 0:batch_shape[1], 0:new_size, 0:new_size]

    return src_maps, dst_maps, mask_borders


def min_max_norm(A):
    shape = A.shape
    A = torch.flatten(A, start_dim=1)
    A -= A.min(1, keepdim=True)[0]
    A /= (A.max(1, keepdim=True)[0] + 1e-6)
    A = torch.reshape(A, shape)

    return A