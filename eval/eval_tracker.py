"""
class to create tracks from corners based on distance in space and time, and use similarity measures for screening
Yuyang
last edit: 20231223
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import cv2
import random
from collections import defaultdict

from model_tracking.tracking_utils import torch_nms, get_topK_max_topk_indices



torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)




class EvalCornerTracker:
    def __init__(self, max_corner_num, device='cpu'):
        """
        time tolerance: (int) time in us
        distance tolerance: (int) distance in pixels
        """
        self.current_corners = torch.zeros((0, 4))
        self.feature_vectors = torch.zeros((0, 128))

        self.current_track_id = 0

        # for vis
        self.traj = defaultdict(list)
        self.updates = defaultdict(int)
        self.colormap = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)[:, 0]
        self.max_diff_t = 10000

        # count track lengths
        self.track_len = 0
        self.num_tracks = 0

        self.max_corner_num = max_corner_num
        self.device = device


    @torch.no_grad()
    def add_and_link_corners(self, heatmap, new_threshold, concat_threshold, time_stamp, event_mask=None):
        '''
        Args:
            heatmap: [B=1, 1, H, W]
            threshold
            time_stamp
            event_mask: [B=1, 1, H, W]
        Returns:
            current_corners: [N, 2]
        '''
        assert heatmap.shape[0] == 1 and heatmap.shape[1] == 1
        nms_kernel_size = 15
        max_pool_kernel_size = 13
        border = 8


        if self.current_corners.shape[0] == 0:  ## init keypoint
            assert self.feature_vectors.shape[0] == 0

            border_mask = torch.zeros_like(heatmap)
            border_mask[:, :, border:-border, border:-border] = 1.0
            if event_mask is not None:
                heatmaps_nms = torch_nms(heatmap*border_mask*event_mask, nms_kernel_size)
            else:
                heatmaps_nms = torch_nms(heatmap*border_mask, nms_kernel_size)
            heatmaps_nms[heatmaps_nms <= new_threshold] = 0
            # y, x = get_topK_max_topk_indices(heatmaps_nms, top_k=self.max_corner_num)
            # y, x = y.squeeze(), x.squeeze()
            y, x = torch.where((heatmaps_nms[0, 0, :, :] > 0).squeeze())
            x, y = self.random_sampling(x, y, target_num=self.max_corner_num)


            corners_xyt = torch.stack([x, y, time_stamp * torch.ones_like(x)], dim=1).to(self.device)    # [N, 3(xyt)]
            self.current_corners = self.add_track_id_to_event_buffer(corners_xyt)   # [N, 4(x, y, t, id)]

            boundary_index = 0   # int
            link_valid_mask = torch.ones(self.current_corners.shape[0]).bool()
            return corners_xyt[:, :2], boundary_index, link_valid_mask
        else:
            cur_corners_loc = self.current_corners[:, :2]   # [N, 2(xy)]

            heatmaps_max_pool = F.max_pool2d(heatmap, kernel_size=max_pool_kernel_size, stride=1, padding=max_pool_kernel_size//2)
            assert heatmaps_max_pool.shape == heatmap.shape
            corners_val = self.extract_N_val(cur_corners_loc.to(heatmap.device),
                                             heatmaps_max_pool.squeeze(0))  # [N]



            link_valid_mask = (corners_val > concat_threshold)  # [N]

            valid_corners = self.current_corners[link_valid_mask]           # [N, 4(x, y, t, id)]
            self.feature_vectors = self.feature_vectors[link_valid_mask]    # [N, 128]

            valid_corners_loc = valid_corners[:, :2]            # [N, 2(xy)]


            ## 2. get new points
            if valid_corners.shape[0] < self.max_corner_num:
                valid_heatmap_mask = torch.zeros_like(heatmap)
                valid_heatmap_mask[:, :, valid_corners_loc[:, 1].long(), valid_corners_loc[:, 0].long()] = 1
                valid_heatmap_mask = F.max_pool2d(valid_heatmap_mask, kernel_size=nms_kernel_size, stride=1, padding=nms_kernel_size // 2)

                heatmap_masked = heatmap * (1-valid_heatmap_mask)

                border_mask = torch.zeros_like(heatmap_masked)
                border_mask[:, :, border:-border, border:-border] = 1.0
                if event_mask is not None:
                    heatmaps_nms = torch_nms(heatmap_masked * border_mask * event_mask, nms_kernel_size)
                else:
                    heatmaps_nms = torch_nms(heatmap_masked * border_mask, nms_kernel_size)
                heatmaps_nms[heatmaps_nms <= new_threshold] = 0
                y, x = torch.where((heatmaps_nms[0, 0, :, :] > 0).squeeze())
                x, y = self.random_sampling(x, y, target_num=(self.max_corner_num-valid_corners.shape[0]))


                corners_xyt = torch.stack([x, y, time_stamp * torch.ones_like(x)], dim=1).to(self.device)  # [N, 3(xyt)]
                new_corners = self.add_track_id_to_event_buffer(corners_xyt)

                self.current_corners = torch.cat([valid_corners, new_corners], dim=0)
            else:
                self.current_corners = valid_corners    # [N, 4]


            boundary_idexes = valid_corners.shape[0]

            return self.current_corners[:, :2], boundary_idexes, link_valid_mask


    @torch.no_grad()
    def update_corners(self,
                       new_corners_loc,
                       new_feature_vectors,
                       time_stamp,
                       image_size):
        '''
        Args:
            new_corners_loc: [B=1, 2(xy), N]
            new_feature_vectors: [B=1, C=128, N]
            time_stamp: float
        '''
        assert self.current_corners.shape[0] == new_corners_loc.shape[-1]
        assert new_corners_loc.shape[-1] == new_feature_vectors.shape[-1]
        new_feature_vectors = new_feature_vectors.squeeze(0).permute(1, 0)  # [B=1, C, N] --> [N, C]


        ## 1.
        H, W =  image_size
        border = 2  # 1  # 0  # 2
        in_W = (new_corners_loc[..., 0, :] > border) * (new_corners_loc[..., 0, :] < W - border)
        in_H = (new_corners_loc[..., 1, :] > border) * (new_corners_loc[..., 1, :] < H - border)
        in_area_mask = (in_W * in_H).squeeze(0)             # [N]

        new_corners_loc = new_corners_loc[..., in_area_mask]   # [B=1, 2(xy), M], M<=N
        new_feature_vectors = new_feature_vectors[in_area_mask, ...]   # [M, 128]


        ## 2.
        new_corners_loc_end = new_corners_loc.squeeze(0).permute(1, 0).to(self.device)      # [M, 2(xy)]
        new_norners_time_end = time_stamp * torch.ones((new_corners_loc_end.shape[0], 1), device=self.device) # [M, 1]
        ids = self.current_corners[:, -1].unsqueeze(-1)     # [N, 1]
        ids = ids[in_area_mask, :]                          # [M, 1]


        self.current_corners = torch.cat([new_corners_loc_end, new_norners_time_end, ids], dim=-1)  # [M, 4]
        self.feature_vectors = new_feature_vectors  # [M, 128]

        return in_area_mask  # [N]



    def random_sampling(self, x, y, target_num):
        N = x.shape[0]
        assert y.shape[0] == N

        if N <= target_num:
            return x, y
        else:
            indices = list(range(N))
            random.shuffle(indices)

            random_indices = indices[:target_num]
            random_indices = torch.tensor(random_indices).long()

            x_sampled = x[random_indices]
            y_sampled = y[random_indices]

            return x_sampled, y_sampled




    def add_track_id_to_event_buffer(self, events, track_id=None):
        """
        Concatenates events with track ids. If track ids are given use them otherwise create new ids
        Args:
            events: torch tensor of shape Nx3 (x, y, t)
            track_id: torch tensor of shape N or Nx1

        Returns:
            Events and ids of shape Nx4
        """
        if track_id is None:
            current_corners = torch.cat(
                [events, torch.arange(self.current_track_id, self.current_track_id + len(events)).view(-1, 1).to(self.device)], axis=1)
            self.current_track_id += len(events)
        else:
            current_corners = torch.cat([events, track_id], axis=1)
        return current_corners


    def save_nn_corners(self, corners, csv_writer, ts):
        """
        Extract corners from the tracker and writes them to a csv
        Args:
            tracker: nearest neighbors tracker class instance
            csv_writer: csv writer
            ts: timestamp of corners

        """
        corners = corners[:, :4].int().cpu().numpy()
        for (x, y, t, id) in corners:
            csv_writer.writerow([x, y, ts, id])



    def extract_N_val(self, points, heatmap):
        '''
        input:
            points: torch tensor of shape (N, 2)
            heatmap: torch tensor of shape (C=1, H, W)
        output:
            val: torch tensor of shape (N)
        '''
        N, _ = points.shape
        C, H, W = heatmap.shape
        assert C==1

        #
        points_normalized = points.clone()
        points_normalized[..., 0] = (points_normalized[..., 0] / (W - 1)) * 2 - 1
        points_normalized[..., 1] = (points_normalized[..., 1] / (H - 1)) * 2 - 1
        points_normalized = points_normalized.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
        print('points_normalized.shape: ', points_normalized.shape)

        #
        vals = F.grid_sample(heatmap.unsqueeze(0), points_normalized, align_corners=True)  # [1, C=1, N, 1]
        vals = vals.squeeze(0).squeeze(0).squeeze(-1)  # [N]

        return vals



    def show(self, corners, frame):
        for x, y, ts, id_ in corners:
            self.traj[int(id_)].append((int(x), int(y)))
            self.updates[int(id_)] = int(ts)

        invalid_ids = []
        for idx, t in self.updates.items():
            diff_t = ts - t
            if diff_t >= self.max_diff_t:
                invalid_ids.append(idx)
        for idx in invalid_ids:
            del self.traj[idx]
            del self.updates[idx]

        # draw on frame the trajectories
        frame = frame.copy()
        for idx, trajectory in self.traj.items():
            color = self.colormap[idx % 255].tolist()
            cv2.circle(frame, trajectory[0], 0, color, 5)
            self.track_len += len(trajectory)
            self.num_tracks += 1
            for i in range(1, len(trajectory)):
                pt1 = trajectory[i - 1]
                pt2 = trajectory[i]
                cv2.line(frame, pt1, pt2, color, 1)
        return frame
