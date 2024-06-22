import torch
import torch.nn as nn
import torch.nn.functional as F
from model_detection.warp_utils import get_image_transform_tensor, scale_homography


def extract_query_vector(point, feature_map):
    '''
    input:
        point: [B, 2(xy)]
        feature_map: [B, C, H, W]
    output:
        query_vector: [B, C, 1, 1]
    '''
    B, C, H, W = feature_map.shape
    point_normalized = point.clone()
    point_normalized[..., 0] = (point_normalized[..., 0] / (W - 1)) * 2 - 1
    point_normalized[..., 1] = (point_normalized[..., 1] / (H - 1)) * 2 - 1
    point_normalized = point_normalized.unsqueeze(1).unsqueeze(1)   # [B, 1, 1, 2]

    query_vector = F.grid_sample(feature_map, point_normalized, align_corners=True)
    return query_vector




def compute_correlation_map(query_vector, feature_map, mode='dot_product'):
    '''
    input:
        query_vector: [B, C, 1, 1]
        feature_map: [B, C, H, W]
        mode: 'dot_product' or 'cosim'
    output:
        correlation_map: [B, 1, H, W]
    '''
    query_vector_shape = query_vector.shape
    assert query_vector_shape[-1] == 1 and query_vector_shape[-2] == 1
    assert query_vector_shape[-3] == feature_map.shape[-3]

    if mode == 'cosim': # norm
        correlation_map = F.cosine_similarity(query_vector, feature_map, dim=-3).unsqueeze(-3)
        return correlation_map
    elif mode == 'dot_product':
        correlation_map = query_vector * feature_map
        correlation_map = torch.sum(correlation_map, dim=-3, keepdim=True)
        return correlation_map
    else:
        raise NotImplementedError



def soft_argmax(heatmap, softmax_temperature=10.0):
    '''
    input:
        heatmap: [B, 1, H, W]
    output:
        point: [B, 2(xy)]
    '''
    B, C, height, width = heatmap.shape

    heatmap_softmax = F.softmax((heatmap*softmax_temperature).view(B, C, -1), dim=-1).view(B, C, height, width)

    # Create grid coordinates for x and y
    y_coordinates, x_coordinates = torch.meshgrid(
        torch.linspace(0, height - 1, height),
        torch.linspace(0, width - 1, width)
    )

    y_coordinates = y_coordinates.to(heatmap_softmax.device)
    x_coordinates = x_coordinates.to(heatmap_softmax.device)

    # Calculate the weighted sum to get the coordinates
    y_coordinate = torch.sum(y_coordinates * heatmap_softmax, dim=(2, 3))
    x_coordinate = torch.sum(x_coordinates * heatmap_softmax, dim=(2, 3))

    coordinates = torch.stack([x_coordinate, y_coordinate], dim=2)

    return coordinates.squeeze(1)



def crop_feature_map(feature_map, center_coords, patch_size):
    '''
    input:
        feature_map: [B, C, H, W]
        center_coords: [B, 2(xy)]
        patch_size: int
    output:
        cropped_patch: [B, C, patch_size, patch_size]
    '''
    assert patch_size % 2 == 1
    radius = patch_size // 2
    B, C, H, W = feature_map.shape

    left = center_coords[..., 0] - radius     # [B]
    top = center_coords[..., 1] - radius

    grid = torch.meshgrid(
        torch.linspace(0, patch_size-1, patch_size),
        torch.linspace(0, patch_size-1, patch_size),
        indexing='xy'
    )

    grid = torch.stack(grid, dim=-1).unsqueeze(0).repeat(B, 1, 1, 1).to(torch.float32).to(feature_map.device)

    grid[..., 0] = grid[..., 0] + left.view(-1, 1, 1)
    grid[..., 1] = grid[..., 1] + top.view(-1, 1, 1)

    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1

    cropped_patch = F.grid_sample(feature_map, grid, align_corners=True)

    return cropped_patch





def torch_nms(input_tensor, kernel_size):
    """
    runs non maximal suppression on square patches of size x size on the two last dimension
    Args:
        input_tensor: torch tensor of shape B, C, H, W
        kernel_size (int): size of the side of the square patch for NMS

    Returns:
        torch tensor where local maximas are unchanged and all other values are -inf
    """
    B, C, H, W = input_tensor.shape
    val, idx = F.max_pool2d(input_tensor, kernel_size=kernel_size, return_indices=True)
    offsets = torch.arange(B * C, device=input_tensor.device) * H * W
    offsets = offsets.repeat_interleave(H // kernel_size).repeat_interleave(W // kernel_size).reshape(B, C,
                                                                                                      H // kernel_size,
                                                                                                      W // kernel_size)
    output_tensor = torch.ones_like(input_tensor) * float("-inf")
    output_tensor.view(-1)[idx + offsets] = val

    return output_tensor





def get_topK_max_topk_indices(image, top_k):
    '''
    input:
        image: [B, 1, H, W]
        top_k: int
    output:
        row_indices: [B, top_k]
        col_indices: [B, top_k]
    '''
    B, C, H, W = image.shape
    assert C == 1
    topk_values, topk_indices = torch.topk(image.view(B, -1), k=top_k, dim=1)

    row_indices = topk_indices // W                   # y, [B, top_k]
    col_indices = topk_indices % W                    # x, [B, top_k]

    return row_indices, col_indices     # y, x




@ torch.no_grad()
def get_topK_gt_trajectory(top_k,
                           pred_heatmaps, query_idx,
                           rotation_vectors, translation_vectors,
                           camera_nts, camera_depths,
                           camera_Ks, camera_Kinvs,
                           origin_sizes, target_size,
                           nms_kernel_size,
                           event_mask=None,
                           ):
    '''
    input:
        top_k: int
        pred_heatmaps: detach [T, B, 10, H, W]
        query_idx: int
        pose_batch: [T, B, C=10, ...]
        target_size: tuple(height, width)
        event_mask: [B, 1, H, W]
    output:
        (1) topk_gt_trajectory: tensor[T=BPTT*10, B, 2(xy), top_k]
    '''
    BPTT_T, B, heatmap_T, H, W = pred_heatmaps.shape
    assert rotation_vectors.shape[:3] == (BPTT_T, B, heatmap_T)
    assert target_size[0] == H and target_size[1] == W

    query_heatmap = pred_heatmaps[0, :, query_idx].unsqueeze(1)                     # [B, 1, H, W]
    query_rotation_vector = rotation_vectors[0, :, query_idx, ...].float()          # [B, 3]
    query_translation_vector = translation_vectors[0, :, query_idx, ...].float()    # [B, 3]


    if event_mask is not None:
        event_mask[event_mask != 0] = 1.0
        query_heatmap = query_heatmap * event_mask  # [B, 1, H, W]


    ## NMS
    query_heatmap = torch_nms(query_heatmap, nms_kernel_size)
    query_heatmap[query_heatmap < 0] = 0



    query_row_indices, query_col_indices = get_topK_max_topk_indices(query_heatmap, top_k)
    z_indices = torch.ones_like(query_col_indices).float()  # z, [B, top_k]

    query_points = torch.stack([query_col_indices, query_row_indices, z_indices], dim=-1)  # [B, top_k, 3]
    query_points = query_points.permute(0, 2, 1).float()    # [B, 3, top_k]


    query_rotation_vector = query_rotation_vector.float().unsqueeze(1).unsqueeze(0).repeat(BPTT_T, 1, heatmap_T, 1).reshape(BPTT_T*B*heatmap_T, 3)          # [BPTT_T, B, heatmap_T, 3]
    query_translation_vector = query_translation_vector.float().unsqueeze(1).unsqueeze(0).repeat(BPTT_T, 1, heatmap_T, 1).reshape(BPTT_T*B*heatmap_T, 3)    # [BPTT_T, B, heatmap_T, 3]
    query_points = query_points.unsqueeze(1).unsqueeze(0).repeat(BPTT_T, 1, heatmap_T, 1, 1).reshape(BPTT_T*B*heatmap_T, 3, top_k)                          # [BPTT_T, B, heatmap_T, 3, top_k]


    target_rotation_vector = rotation_vectors.float().reshape(BPTT_T*B*heatmap_T, 3)           # [BPTT_T*B*heatmap_T, 3]
    target_translation_vector = translation_vectors.float().reshape(BPTT_T*B*heatmap_T, 3)
    target_camera_nt = camera_nts.float().reshape(BPTT_T*B*heatmap_T, 1, 3)
    target_camera_depth = camera_depths.float().reshape(BPTT_T*B*heatmap_T, 1)
    target_camera_K = camera_Ks.float().reshape(BPTT_T*B*heatmap_T, 3, 3)
    target_camera_Kinv = camera_Kinvs.float().reshape(BPTT_T*B*heatmap_T, 3, 3)
    target_origin_size = origin_sizes.float().reshape(BPTT_T*B*heatmap_T, 2)
    target_size = torch.tensor(target_size).unsqueeze(0).repeat(BPTT_T*B*heatmap_T, 1).float().to(query_heatmap.device)  # [BPTT_T*B*heatmap_T, 2]


    homo_query_target = get_image_transform_tensor(query_rotation_vector, query_translation_vector,
                                                   target_rotation_vector, target_translation_vector,
                                                   target_camera_nt, target_camera_depth,
                                                   target_camera_K, target_camera_Kinv)
    homo_query_target_scaled = scale_homography(homo_query_target, target_origin_size, target_size)  # [BPTT_T*B*heatmap_T, 3, 3]


    target_points = torch.matmul(homo_query_target_scaled, query_points) # [BPTT_T*B*heatmap_T, 3, top_k]
    target_z = target_points[:, 2].unsqueeze(1)     # [BPTT_T*B*heatmap_T, 1, top_k]
    target_points = target_points / target_z

    topk_gt_trajectory = target_points[:, :2, :]    # [BPTT_T*B*heatmap_T, 2, top_k]

    topk_gt_trajectory = topk_gt_trajectory.reshape(BPTT_T, B, heatmap_T, 2, top_k).permute(0, 2, 1, 3, 4).reshape(BPTT_T*heatmap_T, B, 2, top_k)   # [T=BPTT*10, B, 2, top_k]


    return topk_gt_trajectory














