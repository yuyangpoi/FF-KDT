import torch
import torch.nn.functional as F
from model_tracking.tracking_utils import compute_correlation_map, soft_argmax



def location_loss(pred_trajectory_each_iter_list,
                  gt_trajectory,
                  image_shape,
                  query_idx):
    '''
    :param pred_trajectory_each_iter_list: list, len: 1+4(init+refine), dshape[T, B, 2(xy), N(point_num)]
    :param gt_trajectory: [T, B, 2(xy), N]
    :param image_shape: (H, W),
    param query_idx：[B, N]
    :return:
        huber_loss
    '''
    iterations = len(pred_trajectory_each_iter_list)
    assert pred_trajectory_each_iter_list[0].shape == gt_trajectory.shape
    H, W = image_shape


    ## Add a mask to points outside the image area
    area_border = 0
    in_W = (gt_trajectory[:, :, 0] >= 0 + area_border) * (gt_trajectory[:, :, 0] <= W - 1 - area_border)
    in_H = (gt_trajectory[:, :, 1] >= 0 + area_border) * (gt_trajectory[:, :, 1] <= H - 1 - area_border)
    in_area_mask = (in_W * in_H).unsqueeze(2).float()   # [T, B, 1] or [T, B, 1, N]


    if query_idx is not None:
        query_idx_repeat = query_idx.unsqueeze(0).unsqueeze(2)                  # [1, B, 1, N]
        in_area_mask_query = torch.gather(in_area_mask, 0, query_idx_repeat)    # [1, B, 1, N]
        in_area_mask = in_area_mask * in_area_mask_query



    weight_sum = 0
    loss_sum = 0
    for i in range(iterations):
        weight = 0.8 ** (iterations-1-i)

        loss_i = F.huber_loss(pred_trajectory_each_iter_list[i], gt_trajectory, reduction='none')   # [T, B, 2(xy)] or [T, B, 2(xy), N]
        loss_i_masked = torch.divide(torch.sum(loss_i * in_area_mask, dim=(2, 3)),
                                     torch.sum(in_area_mask + 1e-6, dim=(2, 3)))

        loss_sum += loss_i_masked.mean()
        weight_sum += weight


    return loss_sum / weight_sum






























