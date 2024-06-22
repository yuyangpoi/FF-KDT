import kornia
from kornia.geometry.transform.imgwarp import homography_warp
import torch
import torch.nn.functional as F
import numpy as np
from numpy.random import uniform
import math
from scipy.stats import truncnorm
import cv2


def generate_homography_tensor(rvec, tvec, nt, depth):
    """
    Generates a single homography

    Args:
        rvec (tensor): rotation vector        [B, 3]
        tvec (tensor): translation vector     [B, 3]
        nt (tensor): normal to camera         [B, 1, 3]
        depth (tensor): depth to camera          [B, 1]
    """
    batch_size = rvec.shape[0]
    R = torch.transpose(kornia.geometry.conversions.angle_axis_to_rotation_matrix(rvec), 1, 2)   # [B, 3, 3]
    H = R - torch.div(torch.matmul(tvec.reshape(batch_size, 3, 1), nt).permute(1, 2, 0), depth.unsqueeze(-1).permute(1, 2, 0)).permute(2, 0, 1)           # [B, 3, 3]
    return H



def generate_image_homography_tensor(rvec, tvec, nt, depth, K, Kinv):
    """
    Generates a single image homography

    Args:
        rvec (tensor): rotation vector          [B, 3]
        tvec (tensor): translation vector       [B, 3]
        nt (tensor): normal to camera           [B, 1, 3]
        depth (tensor): depth to camera         [B, 1]
        K (tensor): intrisic matrix             [B, 3, 3]
        Kinv (tensor): inverse intrinsic matrix [B, 3, 3]
    """
    H = generate_homography_tensor(rvec, tvec, nt, depth)   # [B, 3, 3]
    G = torch.matmul(K, torch.matmul(H, Kinv))  # [B, 3, 3]
    # G /= G[2, 2]
    G = torch.div(G.permute(1, 2, 0), G[:, 2, 2]).permute(2, 0, 1)     # [B, 3, 3]
    return G


def get_image_transform_tensor(rvec1, tvec1, rvec2, tvec2, nt, depth, K, Kinv):
    """
    Get image Homography between 2 poses (includes cam intrinsics)

    Args:
        rvec1 (tensor): rotation vector 1         [B, 3]
        tvec1 (tensor): translation vector 1      [B, 3]
        rvec2 (tensor): rotation vector 2         [B, 3]
        tvec2 (tensor): translation vector 2      [B, 3]
        nt (tensor): plane normal                 [B, 1, 3]
        depth (tensor): depth from camera         [B, 1]
        K (tensor): intrinsic                     [B, 3, 3]
        Kinv (tensor): inverse intrinsic        [B, 3, 3]
    """
    H_0_1 = generate_image_homography_tensor(rvec1, tvec1, nt, depth, K, Kinv)
    H_0_2 = generate_image_homography_tensor(rvec2, tvec2, nt, depth, K, Kinv)
    # print(H_0_1.dtype)
    # H_1_2 = torch.matmul(H_0_2, torch.linalg.inv(H_0_1))  # [B, 3, 3]
    H_1_2 = torch.matmul(H_0_2.float(), torch.linalg.inv(H_0_1.float()))  # [B, 3, 3] TODO: test
    return H_1_2



def scale_homography(homo, origin_size, target_size):
    '''

    :param homo: [B, 3, 3]
    :param origin_size: [B, 2(height, width)]
    :param target_size: [B, 2(height, width)]
    :return:
    '''
    batch_size = homo.shape[0]

    origin_height, origin_width = origin_size[:, 0], origin_size[:, 1]
    target_height, target_width = target_size[:, 0], target_size[:, 1]

    scaling_matrix = torch.tensor([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]], device=homo.device, dtype=torch.float).view(1, 3, 3).repeat(batch_size, 1, 1)

    ## broadcast
    scaling_matrix[:, 0, 0] *= target_width / origin_width
    scaling_matrix[:, 1, 1] *= target_height / origin_height

    homo_scaled_out = torch.matmul(scaling_matrix, torch.matmul(homo, torch.linalg.inv(scaling_matrix)))

    return homo_scaled_out




def warp_perspective_tensor(image_tensor, M_tensor, dsize,
                            mode='bilinear', padding_mode='zeros',
                            normalized_homography=False):
    '''

    :param image_tensor: [B, C, H, W]
    :param M_tensor: homography [B, 3, 3]
    :param dsize: (height, width)
    :param mode:
    :return:
        image_tensor_warpped: [B, C, H, W]
    '''
    return kornia.geometry.transform.warp_perspective(image_tensor, M_tensor, dsize, mode=mode, padding_mode=padding_mode)




def warp_perspective_tensor_by_flow(image_tensor, flow_tensor, dsize,
                            mode='bilinear', padding_mode='zeros'):
    '''

    :param image_tensor: [B, C, H, W]
    :param flow_tensor: [B, 2, H, W]
    :param dsize:
    :param mode:
    :param padding_mode:
    :return:
    '''
    ## TODO
    pass




def gen_random_homography(shape, device):
    default_config = {'perspective': True, 'scaling': True, 'rotation': True, 'translation': True,
                      'n_scales': 5, 'n_angles': 25, 'scaling_amplitude': 0.2, 'perspective_amplitude_x': 0.1,
                      'perspective_amplitude_y': 0.1, 'patch_ratio': 0.5, 'max_angle': math.pi / 2,
                      'allow_artifacts': False, 'translation_overflow': 0.}

    config = default_config

    std_trunc = 2

    # Corners of the input patch
    margin = (1 - config['patch_ratio']) / 2
    pts1 = margin + np.array([[0, 0],
                              [0, config['patch_ratio']],
                              [config['patch_ratio'], config['patch_ratio']],
                              [config['patch_ratio'], 0]])
    pts2 = pts1.copy()

    # Random perspective and affine perturbations
    if config['perspective']:
        if not config['allow_artifacts']:
            perspective_amplitude_x = min(config['perspective_amplitude_x'], margin)
            perspective_amplitude_y = min(config['perspective_amplitude_y'], margin)
        else:
            perspective_amplitude_x = config['perspective_amplitude_x']
            perspective_amplitude_y = config['perspective_amplitude_y']
        perspective_displacement = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_y / 2).rvs(
            1)
        h_displacement_left = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_x / 2).rvs(1)
        h_displacement_right = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_x / 2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if config['scaling']:
        scales = truncnorm(-std_trunc, std_trunc, loc=1, scale=config['scaling_amplitude'] / 2).rvs(
            config['n_scales'])
        # scales = np.random.uniform(0.8, 2, config['n_scales'])
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if config['allow_artifacts']:
            valid = np.arange(config['n_scales'])  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx, :, :]

    # Random translation
    if config['translation']:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if config['allow_artifacts']:
            t_min += config['translation_overflow']
            t_max += config['translation_overflow']
        pts2 += np.array([uniform(-t_min[0], t_max[0], 1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if config['rotation']:
        angles = np.linspace(-config['max_angle'], config['max_angle'], num=config['n_angles'])
        angles = np.concatenate((np.array([0.]), angles), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul((pts2 - center)[np.newaxis, :, :], rot_mat) + center

        if config['allow_artifacts']:
            valid = np.arange(config['n_angles'])  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx, :, :]

    # Rescale to actual size
    shape = np.array(shape[::-1])  # different convention [y, x]
    pts1 *= shape[np.newaxis, :]
    pts2 *= shape[np.newaxis, :]

    # this homography is the same with tf version and this line
    homography = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    homography = torch.tensor(homography, device=device, dtype=torch.float32).unsqueeze(dim=0)

    homography = torch.inverse(homography)  # inverse here to be consistent with tf version

    return homography  # [1,3,3]














