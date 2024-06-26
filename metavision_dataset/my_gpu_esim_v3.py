import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math
from kornia import filters as kornia_filters
from kornia.filters import median_blur

from metavision_core_ml.corner_detection.gpu_corner_esim import collect_target_images
# from metavision_core_ml.video_to_event.gpu_simulator import GPUEventSimulator
from metavision_dataset.simulator.my_gpu_simulator import GPUEventSimulator
from event_utils.event_representation import event_EST



class My_GPUEBSimCorners(object):
    def __init__(self, dataloader, simulator, batch_times,
                 warping_event_volume_depth, exposure_event_volume_depth,
                 randomize_noises, device,
                 number_of_heatmaps, height, width, batch_size, num_workers, needed_number_of_heatmaps,
                 min_blur_ratio, max_blur_ratio, random_gamma_transform, random_pepper_noise):
        self.dataloader = dataloader
        self.simulator = simulator
        self.device = device
        self.batch_times = batch_times
        self.warping_event_volume_depth = warping_event_volume_depth
        self.exposure_event_volume_depth = exposure_event_volume_depth
        self.do_randomize_noises = randomize_noises
        self.number_of_heatmaps = number_of_heatmaps
        assert (self.number_of_heatmaps - 1 + 1) % (needed_number_of_heatmaps - 1) == 0
        self.needed_number_of_heatmaps = needed_number_of_heatmaps


        self.height, self.width = height, width
        self.batch_size = batch_size

        self.num_workers = num_workers



        self.min_blur_ratio = min_blur_ratio
        self.max_blur_ratio = max_blur_ratio
        assert self.max_blur_ratio >= self.min_blur_ratio
        assert self.min_blur_ratio > 0 and self.max_blur_ratio <= 1
        self.blur_ratio_batch = (torch.rand(self.batch_size)*(self.max_blur_ratio-self.min_blur_ratio) + self.min_blur_ratio).to(self.device)


        self.random_gamma_transform = random_gamma_transform
        if self.random_gamma_transform:
            self.gamma_candidate = (0.02, 0.1, 0.5,     0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15,      2, 8, 14)
            self.gamma_batch = torch.as_tensor(np.random.choice(self.gamma_candidate, self.batch_size)).to(self.device)


        self.random_pepper_noise = random_pepper_noise
        if self.random_pepper_noise:
            self.pepper_noise_batch = (torch.rand(self.batch_size) < 0.5).to(self.device)



        self.last_image_end = None
        self.last_corners_map_end = None
        self.last_rotation_vectors_end = None
        self.last_translation_vectors_end = None
        self.last_camera_nts_end = None
        self.last_camera_depths_end = None
        self.last_camera_Ks_end = None
        self.last_camera_Kinvs_end = None
        self.last_origin_sizes_end = None




    def __len__(self):
        return


    def reset(self):
        print('Reset My_GPUEBSimCorners')


    @classmethod
    def from_params(cls, folder,
                    num_workers,
                    batch_size,
                    batch_times,
                    warping_event_volume_depth,
                    exposure_event_volume_depth,
                    height,
                    width,
                    min_frames_per_video,
                    max_frames_per_video,
                    number_of_heatmaps,
                    randomize_noises=False,
                    device='cuda:0',
                    needed_number_of_heatmaps=5,
                    min_blur_ratio=0.2,
                    max_blur_ratio=0.6,
                    random_gamma_transform=True,
                    random_pepper_noise=False,
                    seed=None,

                    max_optical_flow_threshold_random_list=(1, 2, 3)
                    ):
        """
        Creates the simulator from parameters
        Args:
            folder: folder of images
            num_workers: number of workers
            batch_size: size of batch
            batch_times: time dimension per batch
            event_volume_depth: number of channels in event volume
            height: height of images
            width: width of images
            min_frames_per_video: minimum number of frames per video
            max_frames_per_video: maximum number of frames per video
            number_of_heatmaps: number of heatmaps of corners locations returned
            randomize_noises: whether or not to randomize the noise of the simulator
            device: location of data

        Returns:
            GPUEBSimCorners class instantiated
        """
        print('randomize_noises: ', randomize_noises)
        print('random_gamma_transform: ', random_gamma_transform)
        print('random_pepper_noise: ', random_pepper_noise)
        print('number_of_heatmaps: ', number_of_heatmaps)


        number_of_heatmaps_for_new_images = number_of_heatmaps - 1
        print('number_of_heatmaps_for_new_images: ', number_of_heatmaps_for_new_images)

        dataloader = my_make_corner_video_dataset(
            folder, num_workers, batch_size, height, width, min_frames_per_video, max_frames_per_video,
            max_optical_flow_threshold_random_list=max_optical_flow_threshold_random_list,
            number_of_heatmaps=number_of_heatmaps_for_new_images, batch_times=batch_times, seed=seed)
        event_gpu = GPUEventSimulator(batch_size, height, width)
        event_gpu.to(device)
        return cls(dataloader, event_gpu, batch_times,
                   warping_event_volume_depth,
                   exposure_event_volume_depth,
                   randomize_noises, device,
                   number_of_heatmaps_for_new_images, height, width, batch_size, num_workers, needed_number_of_heatmaps,
                   min_blur_ratio, max_blur_ratio, random_gamma_transform, random_pepper_noise)


    def randomize_noises(self, first_times):
        """
        Randomizes noise in the simulator consistent with the batches
        Args:
            first_times: whether or not the video in the batch is new
        """
        batch_size = len(first_times)
        self.simulator.randomize_thresholds(first_times, th_mu_min=0.3,
                                            th_mu_max=0.4, th_std_min=0.08, th_std_max=0.01)
        for i in range(batch_size):
            if first_times[i].item():
                cutoff_rate = 70 if np.random.rand() < 0.1 else 1e6
                leak_rate = 0.1 * 1e-6 if np.random.rand() < 0.1 else 0
                shot_rate = 10 * 1e-6 if np.random.rand() < 0.1 else 0
                refractory_period = np.random.uniform(10, 200)
                self.simulator.cutoff_rates[i] = cutoff_rate
                self.simulator.leak_rates[i] = leak_rate
                self.simulator.shot_rates[i] = shot_rate
                self.simulator.refractory_periods[i] = refractory_period


    def __iter__(self):
        for batch in self.dataloader:
            gray_images = batch['images'].squeeze(0).to(self.device)    # torch.Size([360, 480, batchsize*T]), T=number_of_heatmaps*num_tbins
            # print('gray_images.shape: ', gray_images.shape)
            first_times = batch['first_times'].to(self.device)
            timestamps = batch['timestamps'].to(self.device)
            video_len = batch["video_len"].to(self.device)
            target_indices = batch['target_indices']
            prev_ts = self.simulator.prev_image_ts.clone()
            prev_ts = prev_ts * (1 - first_times) + timestamps[:, 0] * first_times

            # homos = batch['homos'].to(self.device)  # homography between two consecutive frames
            rotation_vectors = batch['rotation_vectors'].to(self.device)        # torch.Size([3, batchsize*T]), T=number_of_heatmaps*num_tbins
            translation_vectors = batch['translation_vectors'].to(self.device)  # torch.Size([3, batchsize*T]), T=number_of_heatmaps*num_tbins
            camera_nts = batch['camera_nts'].to(self.device)                    # torch.Size([1, 3, batchsize*T]), T=number_of_heatmaps*num_tbins
            camera_depths = batch['camera_depths'].to(self.device)              # torch.Size([1, batchsize*T]), T=number_of_heatmaps*num_tbins
            camera_Ks = batch['camera_Ks'].to(self.device)                      # torch.Size([3, 3, batchsize*T]), T=number_of_heatmaps*num_tbins
            camera_Kinvs = batch['camera_Kinvs'].to(self.device)                # torch.Size([3, 3, batchsize*T]), T=number_of_heatmaps*num_tbins
            origin_sizes = batch['origin_sizes'].to(self.device)                # torch.Size([2, batchsize*T]), T=number_of_heatmaps*num_tbins


            if self.do_randomize_noises:
                self.randomize_noises(batch['first_times'])


            log_images = self.simulator.dynamic_moving_average(gray_images, video_len, timestamps, first_times)


            target_images, target_times = collect_target_images(gray_images, timestamps, batch['video_len'], target_indices, self.number_of_heatmaps)



            # target_homos, _ = collect_target_images(homos, timestamps, batch['video_len'], target_indices, self.number_of_heatmaps) # [T, B, number_of_heatmaps, 3, 3]
            target_rotation_vectors, _ = collect_target_images(rotation_vectors.unsqueeze(0), timestamps, batch['video_len'], target_indices, self.number_of_heatmaps)
            target_rotation_vectors = target_rotation_vectors.squeeze(-2)           # [T, B, number_of_heatmaps, 3]
            target_translation_vectors, _ = collect_target_images(translation_vectors.unsqueeze(0), timestamps, batch['video_len'], target_indices, self.number_of_heatmaps)
            target_translation_vectors = target_translation_vectors.squeeze(-2)     # [T, B, number_of_heatmaps, 3]
            target_camera_nts, _ = collect_target_images(camera_nts, timestamps, batch['video_len'], target_indices, self.number_of_heatmaps)       # [T, B, number_of_heatmaps, 1, 3]
            target_camera_depths, _ = collect_target_images(camera_depths.unsqueeze(0), timestamps, batch['video_len'], target_indices, self.number_of_heatmaps)
            target_camera_depths = target_camera_depths.squeeze(-2)                 # [T, B, number_of_heatmaps, 1]
            target_camera_Ks, _ = collect_target_images(camera_Ks, timestamps, batch['video_len'], target_indices, self.number_of_heatmaps)         # [T, B, number_of_heatmaps, 3, 3]
            target_camera_Kinvs, _ = collect_target_images(camera_Kinvs, timestamps, batch['video_len'], target_indices, self.number_of_heatmaps)   # [T, B, number_of_heatmaps, 3, 3]
            target_origin_sizes, _ = collect_target_images(origin_sizes.unsqueeze(0), timestamps, batch['video_len'], target_indices, self.number_of_heatmaps)
            target_origin_sizes = target_origin_sizes.squeeze(-2)   # [T, B, number_of_heatmaps, 2]



            # print('prev_ts: ', prev_ts)
            # print('target_times: ', target_times)           # [B, BPTT_t]
            # print('timestamps.shape: ', timestamps.shape)   # [B, BPTT_t*40]
            # print('timestamps: ', timestamps)
            # print('timestamps[:, 1:]-timestamps[:, :-1]: ', timestamps[:, 1:]-timestamps[:, :-1])
            # print('self.needed_number_of_heatmaps: ', self.needed_number_of_heatmaps)
            blur_ratio_batch = self.update_blur_ratio_batch(first_times)
            raw_event_stream = self.simulator.get_events(log_images, video_len, timestamps, first_times)   # [event_num, 5(batch_index, x, y, polarity, timestamp)]
            frame_interval_EST, exposure_EST = self.get_EST_sequence(raw_event_stream, prev_ts, target_times,
                                                                     self.warping_event_volume_depth, self.exposure_event_volume_depth,
                                                                     blur_ratio_batch,
                                                                     timestamps, self.needed_number_of_heatmaps)


            reset = 1 - first_times[:, None, None, None]



            target_rotation_vectors, self.last_rotation_vectors_end = self.concat_last_corners_map_end(target_rotation_vectors, self.last_rotation_vectors_end, first_times)
            target_translation_vectors, self.last_translation_vectors_end = self.concat_last_corners_map_end(target_translation_vectors, self.last_translation_vectors_end, first_times)
            target_camera_nts, self.last_camera_nts_end = self.concat_last_corners_map_end(target_camera_nts, self.last_camera_nts_end, first_times)
            target_camera_depths, self.last_camera_depths_end = self.concat_last_corners_map_end(target_camera_depths, self.last_camera_depths_end, first_times)
            target_camera_Ks, self.last_camera_Ks_end = self.concat_last_corners_map_end(target_camera_Ks, self.last_camera_Ks_end, first_times)
            target_camera_Kinvs, self.last_camera_Kinvs_end = self.concat_last_corners_map_end(target_camera_Kinvs, self.last_camera_Kinvs_end, first_times)
            target_origin_sizes, self.last_origin_sizes_end = self.concat_last_corners_map_end(target_origin_sizes, self.last_origin_sizes_end, first_times)


            needed_rotation_vectors, needed_translation_vectors, \
            needed_camera_nts, needed_camera_depths, \
            needed_camera_Ks, needed_camera_Kinvs,\
            needed_origin_sizes = self.get_needed_homos(target_rotation_vectors, target_translation_vectors,
                                                 target_camera_nts, target_camera_depths,
                                                 target_camera_Ks, target_camera_Kinvs,
                                                 target_origin_sizes,
                                                 num=self.needed_number_of_heatmaps)



            target_images, self.last_image_end = self.concat_last_corners_map_end(target_images, self.last_image_end, first_times)
            # needed_images = self.get_needed_corners_map(target_images, self.needed_number_of_heatmaps)  # TODO: test

            if self.random_gamma_transform:
                gamma_batch = self.update_gamma_batch(first_times)
                blurred_images = self.gen_blurred_image(target_images, blur_ratio_batch, gamma_batch)
            else:
                blurred_images = self.gen_blurred_image(target_images, blur_ratio_batch, gamma_batch=None)

            blurred_images = blurred_images.to(torch.uint8).float()

            clear_images = target_images[:, :, 0].unsqueeze(2)

            event_binary_mask, frame_binary_mask = self.get_binary_mask(target_images[:, :, 0].unsqueeze(2), frame_interval_EST)

            if self.random_pepper_noise:
                pepper_noise_batch = self.update_pepper_noise_batch(first_times)
                frame_interval_EST = self.add_pepper_noise(frame_interval_EST, pepper_noise_batch)
                exposure_EST = self.add_pepper_noise(exposure_EST, pepper_noise_batch)




            out_batch = {
                # 'images': target_images,        # torch.Size([time_num_tbins, batchsize, number_of_heatmaps, H, W])
                # 'images_needed': needed_images, # TODO: test
                # 'raw_corners_map': target_corners_map, # torch.Size([time_num_tbins, batchsize, number_of_heatmaps, H, W])


                'frame_interval_events': frame_interval_EST, # torch.Size([time_num_tbins, batchsize, C*2, H, W])
                'exposure_events': exposure_EST,  # torch.Size([time_num_tbins, batchsize, C*2, H, W])

                'reset': reset,

                'blurred_images': blurred_images,   # torch.Size([time_num_tbins, batchsize, 1, H, W]), 0~255
                'clear_images': clear_images,       # torch.Size([time_num_tbins, batchsize, 1, H, W]), 0~255

                'rotation_vectors': needed_rotation_vectors,        # torch.Size([time_num_tbins, batchsize, needed_corners_map, 3])
                'translation_vectors': needed_translation_vectors,  # torch.Size([time_num_tbins, batchsize, needed_corners_map, 3])
                'camera_nts': needed_camera_nts,                    # torch.Size([time_num_tbins, batchsize, needed_corners_map, 1, 3])
                'camera_depths': needed_camera_depths,              # torch.Size([time_num_tbins, batchsize, needed_corners_map, 1])
                'camera_Ks': needed_camera_Ks,                      # torch.Size([time_num_tbins, batchsize, needed_corners_map, 3, 3])
                'camera_Kinvs': needed_camera_Kinvs,                # torch.Size([time_num_tbins, batchsize, needed_corners_map, 3, 3])
                'origin_sizes': needed_origin_sizes,                # torch.Size([time_num_tbins, batchsize, needed_corners_map, 2])

                'event_binary_masks': event_binary_mask,            # torch.Size([time_num_tbins, batchsize, 1, H, W])
                'frame_binary_masks': frame_binary_mask              # torch.Size([time_num_tbins, batchsize, 1, H, W])

            }

            yield out_batch




    def get_EST_sequence(self, event_stream, begin_times, target_times,
                         warping_n_bins, exposure_n_bins,
                         blur_ratio_batch,
                         all_timestamps, sub_num):
        B, T = target_times.shape
        assert blur_ratio_batch.shape[0] == B
        # print('B, T: ', B, T)
        # print('target_times: ', target_times)

        frame_interval_EST_list_B = []
        exposure_EST_list_B = []
        for bb in range(B):
            index_bb = (event_stream[:, 0] == bb)
            event_stream_bb = event_stream[index_bb]    # [events_num, 5]
            target_times_bb = target_times[bb]
            blur_ratio_bb = blur_ratio_batch[bb]

            all_timestamps_bb = all_timestamps[bb]

            frame_interval_EST_list_T = []
            exposure_EST_list_T = []

            last_target_times_bb_tt = begin_times[bb]
            # print('torch.min(event_stream_bb[:, -1]): ', torch.min(event_stream_bb[:, -1]))
            for tt in range(T):
                target_times_bb_tt = target_times_bb[tt]
                # print('\nbb: {}, tt:{}, target_times_bb_tt: {}'.format(bb, tt, target_times_bb_tt))
                # print('last_time: ', last_target_times_bb_tt)
                # print('cur_time: ', target_times_bb_tt)
                # print('blur_ratio_bb: ', blur_ratio_bb)

                frame_interval_index_bb_tt = (event_stream_bb[:, -1] >= last_target_times_bb_tt) * (event_stream_bb[:, -1] <= target_times_bb_tt)
                frame_interval_event_stream_bb_tt = event_stream_bb[frame_interval_index_bb_tt, 1:]  # [events_num, 4]



                if tt == 0:
                    all_timestamps_bb_tt_index = (all_timestamps_bb>=last_target_times_bb_tt) * (all_timestamps_bb<=target_times_bb_tt)
                else:
                    all_timestamps_bb_tt_index = (all_timestamps_bb > last_target_times_bb_tt) * (all_timestamps_bb <= target_times_bb_tt)
                all_timestamps_bb_tt = all_timestamps_bb[all_timestamps_bb_tt_index]    # [40]
                all_timestamps_bb_tt = torch.cat([torch.tensor([last_target_times_bb_tt]).to(all_timestamps_bb_tt.device), all_timestamps_bb_tt])   # [41]
                # print('222 all_timestamps_bb_tt.shape: ', all_timestamps_bb_tt.shape)
                # print('222 all_timestamps_bb_tt: ', all_timestamps_bb_tt)
                sub_idx_interval = (all_timestamps_bb_tt.shape[0] - 1) // (sub_num - 1)
                sub_timestamps_bb_tt = all_timestamps_bb_tt[0: all_timestamps_bb_tt.shape[0]:sub_idx_interval]
                # print('sub_timestamps_bb_tt: ', sub_timestamps_bb_tt)


                frame_interval_EST_list_T_sub = []
                zero_event_time_0 = sub_timestamps_bb_tt[0]
                zero_event_time_1 = (sub_timestamps_bb_tt[1] - zero_event_time_0)*0.005 + zero_event_time_0
                zero_event_index_bb_tt = (frame_interval_event_stream_bb_tt[:, -1] >= zero_event_time_0) \
                                         * (frame_interval_event_stream_bb_tt[:, -1] <= zero_event_time_1)
                zero_event_stream_bb_tt = frame_interval_event_stream_bb_tt[zero_event_index_bb_tt]
                zero_EST_bb_tt = event_EST(events=zero_event_stream_bb_tt,
                                             height=self.height,
                                             width=self.width,
                                             start_times=zero_event_time_0,
                                             durations=zero_event_time_1 - zero_event_time_0,
                                             nbins=warping_n_bins)
                frame_interval_EST_list_T_sub.append(zero_EST_bb_tt)


                for sub_i in range(sub_num-1):
                    #### time_0, time_1 = sub_timestamps_bb_tt[sub_i], sub_timestamps_bb_tt[sub_i+1]
                    time_0 = sub_timestamps_bb_tt[0]
                    time_1 = sub_timestamps_bb_tt[sub_i+1]
                    frame_interval_index_bb_tt_sub = (frame_interval_event_stream_bb_tt[:, -1] >= time_0) * (frame_interval_event_stream_bb_tt[:, -1] <= time_1)
                    frame_interval_event_stream_bb_tt_sub = frame_interval_event_stream_bb_tt[frame_interval_index_bb_tt_sub]   # [events_num, 4]
                    # print('sub_i: ', sub_i)
                    # print('time_0, time_1: ', time_0, time_1)
                    frame_interval_EST_bb_tt_sub = event_EST(events=frame_interval_event_stream_bb_tt_sub,
                                                         height=self.height,
                                                         width=self.width,
                                                         start_times=time_0,
                                                         durations=time_1-time_0,
                                                         nbins=warping_n_bins)
                    frame_interval_EST_list_T_sub.append(frame_interval_EST_bb_tt_sub)
                frame_interval_EST_list_T.append(torch.cat(frame_interval_EST_list_T_sub, dim=-3))  # [C*(4+1), H, W]



                blur_num = math.ceil(all_timestamps_bb_tt.shape[0] * blur_ratio_bb)
                exposure_end_time = all_timestamps_bb_tt[:blur_num][-1]
                # print('exposure_end_time: ', exposure_end_time)

                exposure_index_bb_tt = (frame_interval_event_stream_bb_tt[:, -1] >= last_target_times_bb_tt) * (frame_interval_event_stream_bb_tt[:, -1] <= exposure_end_time)
                exposure_event_stream_bb_tt = frame_interval_event_stream_bb_tt[exposure_index_bb_tt]   # [events_num, 4]
                # print('exposure_event_stream_bb_tt.shape: ', exposure_event_stream_bb_tt.shape)
                exposure_EST_bb_tt = event_EST(events=exposure_event_stream_bb_tt,
                                                     height=self.height,
                                                     width=self.width,
                                                     start_times=last_target_times_bb_tt,
                                                     durations=exposure_end_time - last_target_times_bb_tt,
                                                     nbins=exposure_n_bins)
                exposure_EST_list_T.append(exposure_EST_bb_tt)

                last_target_times_bb_tt = target_times_bb_tt



            frame_interval_EST_list_B.append(torch.stack(frame_interval_EST_list_T, dim=0)) # [T, C*5, H, W]
            exposure_EST_list_B.append(torch.stack(exposure_EST_list_T, dim=0))             # [T, C, H, W]

        frame_interval_EST = torch.stack(frame_interval_EST_list_B, dim=1)  # [T, B, C*cout, H, W]
        exposure_EST = torch.stack(exposure_EST_list_B, dim=1)              # [T, B, C, H, W]
        return frame_interval_EST, exposure_EST



    def get_binary_mask(self, images, event):
        '''
        Args:
            images: [T, B, 1, H, W]
            event: [T, B, C, H, W]
        Returns:
            event_binary_mask: [T, B, 1, H, W]
            frame_binary_mask: [T, B, 1, H, W]
        '''
        T, B, _, H, W = images.shape

        ## 1. event binary mask
        event_mask = torch.sum(torch.abs(event), dim=2, keepdim=True)  # [T, B, 1, H, W]
        event_mask = (event_mask != 0).reshape(T*B, 1, H, W).float()
        event_mask = median_blur(event_mask, (5, 5))    # .reshape(T, B, 1, H, W).bool()   # [T, B, 1, H, W]
        event_mask = F.max_pool2d(event_mask, kernel_size=3, stride=1, padding=1).reshape(T, B, 1, H, W).bool()


        ## 2. frame binary mask
        images = images.reshape(T*B, 1, H, W)
        image_padded = F.pad(images.float(), (1, 1, 1, 1), mode='reflect')
        image_filtered = torch.nn.functional.avg_pool2d(image_padded, kernel_size=3, stride=1, padding=0)

        gradient = kornia_filters.spatial_gradient(image_filtered)
        gradient_intensity = torch.square(gradient[:, :, 0]) + torch.square(gradient[:, :, 1])

        frame_mask = (gradient_intensity > 64).float()
        frame_mask = median_blur(frame_mask, (5, 5))
        frame_mask = F.max_pool2d(frame_mask, kernel_size=3, stride=1, padding=1).reshape(T, B, 1, H, W).bool()

        return event_mask, frame_mask



    def add_pepper_noise(self, event_representation, pepper_noise_batch):
        '''
        input:
            event_representation: torch.Size([time_num_tbins, batchsize, number_of_heatmaps, H, W])
            pepper_noise_batch: [batchsize]
        output:
            event_representation_noised: torch.Size([time_num_tbins, batchsize, number_of_heatmaps, H, W])
        '''
        clean_event_representation = event_representation[:, ~pepper_noise_batch]

        device = event_representation.device
        noise_mask = torch.rand(event_representation.shape, device=device) < 0.0004
        noise_val = torch.rand(event_representation.shape, device=device) * 0.8

        non_zero_mask = (event_representation == 0)
        event_representation[noise_mask & non_zero_mask] += noise_val[noise_mask & non_zero_mask]

        event_representation[:, ~pepper_noise_batch] = clean_event_representation

        return event_representation


    def concat_last_corners_map_end(self, image, last_image_end, first_times):
        '''
        :param image: image or corners map or transform matrix [T, B, C, ...]
        :param last_image_end: last image end or last corners end [B, ...], ([T=1, B, C=1, ...])
        :param first_times: 1 表示新的序列 [B]
        :return:
            [T, B, C+1, ...]
        '''
        T, B, C = image.shape[:3]
        output = []
        new_last_image_end = []
        for bb in range(B):
            image_bb = image[:, bb]     # [T, C, ...]
            ## old seq
            if first_times[bb] == 0:
                last_image_end_bb = last_image_end[bb]
            ## new seq
            elif first_times[bb] == 1:
                last_image_end_bb = image_bb[0, 0]
            last_images_bb = image_bb[:T-1, -1]
            last_images_bb = torch.cat([last_image_end_bb.unsqueeze(0), last_images_bb], dim=0)

            output_bb = torch.cat([last_images_bb.unsqueeze(1), image_bb], 1)
            output.append(output_bb)

            new_last_image_end.append(image_bb[-1, -1])
        output = torch.stack(output, dim=1)
        new_last_image_end = torch.stack(new_last_image_end, dim=0) # [B, ...]
        return output, new_last_image_end


    def get_needed_corners_map(self, raw_corners_map, num=10):
        T, B, C, H, W = raw_corners_map.shape
        assert (C-1) % (num-1) == 0
        interval = (C-1) // (num-1)

        needed_corners_map = raw_corners_map[:, :, 0:C:interval, :, :]
        return needed_corners_map


    def get_needed_homos(self, rotation_vectors, translation_vectors,
                         camera_nts, camera_depths,
                         camera_Ks, camera_Kinvs,
                         orgin_sizes,
                         num=10):
        '''

        :param rotation_vectors: [T, B, C, 3]
        :param translation_vectors: [T, B, C, 3]
        :param camera_nts: [T, B, C, 1, 3]
        :param camera_depths: [T, B, C, 1]
        :param camera_Ks: [T, B, C, 3, 3]
        :param camera_Kinvs: [T, B, C, 3, 3]
        :param orgin_sizes: [T, B, C, 2]
        :param num: int
        :return:
        '''
        T, B, C = rotation_vectors.shape[:3]
        assert (C - 1) % (num - 1) == 0
        interval = (C - 1) // (num - 1)

        needed_rotation_vectors = rotation_vectors[:, :, 0:C:interval]
        needed_translation_vectors = translation_vectors[:, :, 0:C:interval]
        needed_camera_nts = camera_nts[:, :, 0:C:interval]
        needed_camera_depths = camera_depths[:, :, 0:C:interval]
        needed_camera_Ks = camera_Ks[:, :, 0:C:interval]
        needed_camera_Kinvs = camera_Kinvs[:, :, 0:C:interval]
        needed_orgin_sizes = orgin_sizes[:, :, 0:C:interval]

        return needed_rotation_vectors, needed_translation_vectors, \
               needed_camera_nts, needed_camera_depths, \
               needed_camera_Ks, needed_camera_Kinvs, \
               needed_orgin_sizes



    def gamma_transform(self, image, gamma):
        if gamma >= 1.0:
            return ((image.float() / 255.) ** gamma) * 255
        else:
            image_transformed = (((image+1).float() / 255.) ** gamma) * 255
            image_transformed[image_transformed>255] = 255
            return image_transformed


    def update_blur_ratio_batch(self, first_time):
        if torch.sum(first_time) > 0:
            new_ratio_batch = (torch.rand(self.batch_size)*(self.max_blur_ratio-self.min_blur_ratio) + self.min_blur_ratio).to(self.device)
            self.blur_ratio_batch = (1 - first_time) * self.blur_ratio_batch + first_time * new_ratio_batch
        return self.blur_ratio_batch

    def update_gamma_batch(self, first_time):
        if torch.sum(first_time) > 0:
            new_gamma_batch = torch.as_tensor(np.random.choice(self.gamma_candidate, self.batch_size)).to(self.device)
            self.gamma_batch = (1 - first_time) * self.gamma_batch + first_time * new_gamma_batch
        return self.gamma_batch

    def update_pepper_noise_batch(self, first_time):
        if torch.sum(first_time) > 0:
            new_papper_noise_batch = (torch.rand(self.batch_size) < 0.5).to(self.device)
            self.pepper_noise_batch = ((1 - first_time) * self.pepper_noise_batch + first_time * new_papper_noise_batch).bool()
        return self.pepper_noise_batch


    def gen_blurred_image(self, image_batch, blur_ratio_batch, gamma_batch):
        '''
        input:
            image_batch: torch.Size([time_num_tbins, batchsize, number_of_heatmaps, H, W])
            blur_ratio_batch: torch.Size([batchsize])
            gamma_batch: torch.Size([batchsize])
        output:
            blurred_image: torch.Size([time_num_tbins, batchsize, 1, H, W])
        '''
        _, batchsize, number_of_heatmaps, _, _ = image_batch.shape

        blurred_images_list = []
        for i in range(batchsize):
            blur_num = math.ceil(number_of_heatmaps * blur_ratio_batch[i])
            blurred_image = torch.mean(image_batch[:, i, :blur_num].float(), dim=1, keepdim=True)

            if gamma_batch is not None:
                blurred_image_gamma_transformed = self.gamma_transform(blurred_image, gamma_batch[i])
                blurred_images_list.append(blurred_image_gamma_transformed)
            else:
                blurred_images_list.append(blurred_image)

        blurred_images = torch.stack(blurred_images_list, dim=1)
        return blurred_images


from .my_scheduling import build_metadata
from metavision_core_ml.data.stream_dataloader import StreamDataset, StreamDataLoader
def pad_collate_fn(data_list):
    """
    Here we pad with last image/ timestamp to get a contiguous batch
    """
    # images, corners, timestamps, target_indices, first_times, homos = zip(*data_list)
    images, timestamps, target_indices, first_times, \
    rotation_vectors, translation_vectors, \
    camera_nts, camera_depths, camera_Ks, camera_Kinvs, \
    origin_height, origin_width = zip(*data_list)

    # print('type(images): ', type(images))   # tuple, len=batch_size
    video_len = [item.shape[-1] for item in images]
    max_len = max([item.shape[-1] for item in images])
    b = len(images)
    c, h, w = images[0].shape[1:-1]
    out_images = torch.zeros((c, h, w, sum(video_len)), dtype=images[0].dtype)  # [C=1, H, W, batchsize*T(T=230)]
    out_timestamps = torch.zeros((b, max_len), dtype=timestamps[0].dtype)
    target_indices = torch.cat(target_indices).int()


    # out_homos = torch.zeros((3, 3, sum(video_len)), dtype=torch.float)
    out_rotation_vectors = torch.zeros((3, sum(video_len)), dtype=rotation_vectors[0].dtype)        # [3, batchsize*T(T=230)]
    out_translation_vectors = torch.zeros((3, sum(video_len)), dtype=translation_vectors[0].dtype)  # [3, batchsize*T(T=230)]
    out_camera_nts = torch.zeros((1, 3, sum(video_len)), dtype=camera_nts[0].dtype)                 # [1, 3, batchsize*T(T=230)]
    out_camera_depths = torch.zeros((1, sum(video_len)), dtype=camera_depths[0].dtype)              # [1, batchsize*T(T=230)]
    out_camera_Ks = torch.zeros((3, 3, sum(video_len)), dtype=camera_Ks[0].dtype)                   # [3, 3, batchsize*T(T=230)]
    out_camera_Kinvs = torch.zeros((3, 3, sum(video_len)), dtype=camera_Kinvs[0].dtype)             # [3, 3, batchsize*T(T=230)]
    out_origin_sizes = torch.zeros((2, sum(video_len)), dtype=origin_height[0].dtype)              # [2, batchsize*T(T=230)]



    current_ind = 0
    for i in range(b):
        video = images[i]
        ilen = video.shape[-1]
        out_images[..., current_ind: current_ind + ilen] = video

        # out_homos[..., current_ind: current_ind + ilen] = homos[i]
        out_rotation_vectors[..., current_ind: current_ind + ilen] = rotation_vectors[i]
        out_translation_vectors[..., current_ind: current_ind + ilen] = translation_vectors[i]
        out_camera_nts[..., current_ind: current_ind + ilen] = camera_nts[i]
        out_camera_depths[..., current_ind: current_ind + ilen] = camera_depths[i]
        out_camera_Ks[..., current_ind: current_ind + ilen] = camera_Ks[i]
        out_camera_Kinvs[..., current_ind: current_ind + ilen] = camera_Kinvs[i]
        out_origin_sizes[0, current_ind: current_ind + ilen] = origin_height[i]
        out_origin_sizes[1, current_ind: current_ind + ilen] = origin_width[i]


        current_ind += ilen
        out_timestamps[i, :ilen] = timestamps[i]
        out_timestamps[i, ilen:] = timestamps[i][:, ilen - 1:].unsqueeze(1)

    first_times = torch.FloatTensor(first_times)
    return {'images': out_images,
            'timestamps': out_timestamps,
            'target_indices': target_indices,
            'first_times': first_times,
            'video_len': torch.tensor(video_len, dtype=torch.int32),

            # 'homos': out_homos
            'rotation_vectors': out_rotation_vectors,
            'translation_vectors': out_translation_vectors,
            'camera_nts': out_camera_nts,
            'camera_depths': out_camera_depths,
            'camera_Ks': out_camera_Ks,
            'camera_Kinvs': out_camera_Kinvs,
            'origin_sizes': out_origin_sizes

            }
def my_make_corner_video_dataset(path, num_workers, batch_size, height, width, min_length, max_length,
                                max_optical_flow_threshold_random_list,
                              number_of_heatmaps=10, rgb=False, seed=None, batch_times=1):
    """
    Makes a video/ moving picture dataset.

    Args:
        path (str): folder to dataset
        batch_size (int): number of video clips / batch
        height (int): height
        width (int): width
        min_length (int): min length of video
        max_length (int): max length of video
        mode (str): 'frames' or 'delta_t'
        num_tbins (int): number of bins in event volume
        number_of_heatmaps (int): number of corner heatmaps predicted by the network
        rgb (bool): retrieve frames in rgb
        seed (int): seed for randomness
        batch_times (int): number of time steps in training sequence
    """
    metadata = build_metadata(path, min_length, max_length, denominator=number_of_heatmaps*batch_times)
    print('scheduled streams: ', len(metadata))

    def iterator_fun(metadata):
        return My_CornerVideoDatasetIterator(
            metadata, height, width, rgb=rgb,
            max_optical_flow_threshold_random_list=max_optical_flow_threshold_random_list,
            number_of_heatmaps=number_of_heatmaps, batch_times=batch_times)
    dataset = StreamDataset(metadata, iterator_fun, batch_size, "data", None, seed)
    dataloader = StreamDataLoader(dataset, num_workers, pad_collate_fn)

    return dataloader



class My_CornerVideoDatasetIterator(object):
    """
    Dataset Iterator streaming images, timestamps and corners

    Args:
        metadata (object): path to picture or video
        height (int): height of input images / video clip
        width (int): width of input images / video clip
        rgb (bool): stream rgb videos
        number_of_heatmaps (int): The number of heatmaps containing corner locations
        batch_times (int): number of timesteps of training sequences
    """

    def __init__(self, metadata, height, width, rgb,
                 max_optical_flow_threshold_random_list,
                 number_of_heatmaps=10, batch_times=1):
        self.image_stream = My_CornerPlanarMotionStream(metadata.path, height, width,
                                                        max_optical_flow_threshold_random_list,
                                                        len(metadata), rgb=rgb)
        self.height = height
        self.width = width
        self.rgb = rgb
        self.metadata = metadata
        self.mode = 'frames'
        self.number_of_heatmaps = number_of_heatmaps
        self.batch_times = batch_times

    def __iter__(self):
        img_out = []
        times = []
        target_indices = []
        first_time = True
        last_time = None

        # homo_out = []
        rotation_vector_out = []
        translation_vector_out = []
        camera_nt_out = []
        camera_depth_out = []
        camera_K_out = []
        camera_Kinv_out = []
        origin_height_out = []
        origin_width_out = []


        for i, (img, ts, pose_dict) in enumerate(self.image_stream):
            if img.ndim == 3:
                img = np.moveaxis(img, 2, 0)
            else:
                img = img[None]

            img_out.append(img[None, ..., None])  # B,C,H,W,T or B,H,W,T

            # homo_out.append(homo[None, ..., None])  # [B=1, 3, 3, T=1]
            rotation_vector = pose_dict['rotation_vector']          # [3, ]
            translation_vector = pose_dict['translation_vector']    # [3, ]
            camera_nt = pose_dict['nt']                             # [1, 3]
            camera_depth = pose_dict['depth']                       # float
            camera_K = pose_dict['K']                               # [3, 3]
            camera_Kinv = pose_dict['Kinv']                         # [3, 3]
            origin_height = pose_dict['origin_height']              # float
            origin_width = pose_dict['origin_width']                # float

            rotation_vector_out.append(rotation_vector[None, ..., None])        # [B=1, 3, T=1]
            translation_vector_out.append(translation_vector[None, ..., None])  # [B=1, 3, T=1]
            camera_nt_out.append(camera_nt[None, ..., None])                    # [B=1, 1, 3, T=1]
            camera_depth_out.append(np.array(camera_depth)[None][None, ..., None])    # [B=1, 1, T=1]
            camera_K_out.append(camera_K[None, ..., None])                      # [B=1, 3, 3, T=1]
            camera_Kinv_out.append(camera_Kinv[None, ..., None])                # [B=1, 3, 3, T=1]
            origin_height_out.append(np.array(origin_height)[None][None, ..., None]) # [B=1, 1, T=1]
            origin_width_out.append(np.array(origin_width)[None][None, ..., None])  # [B=1, 1, T=1]


            times.append(ts)
            if last_time is None:
                last_time = ts

            if len(img_out) % self.number_of_heatmaps == 0:
                target_indices.append(len(img_out)-1)

            if len(img_out) == (self.batch_times*self.number_of_heatmaps):
                image_sequence = torch.from_numpy(np.concatenate(img_out, axis=-1))
                timestamps = torch.FloatTensor(times)[None, :]  # B,T


                # homo_sequence = torch.from_numpy(np.concatenate(homo_out, axis=-1)) # [B=1, 3, 3, T(=230)]
                # print('homo_sequence.shape: ', homo_sequence.shape)
                rotation_vector_sequence = torch.from_numpy(np.concatenate(rotation_vector_out, axis=-1))           # [B=1, 3, T(=230)]
                translation_vector_sequence = torch.from_numpy(np.concatenate(translation_vector_out, axis=-1))     # [B=1, 3, T(=230)]
                camera_nt_sequence = torch.from_numpy(np.concatenate(camera_nt_out, axis=-1))                       # [B=1, 1, 3, T(=230)]
                camera_depth_sequence = torch.from_numpy(np.concatenate(camera_depth_out, axis=-1))                 # [B=1, 1, T(=230)]
                camera_K_sequence = torch.from_numpy(np.concatenate(camera_K_out, axis=-1))                         # [B=1, 3, 3, T(=230)]
                camera_Kinv_sequence = torch.from_numpy(np.concatenate(camera_Kinv_out, axis=-1))                   # [B=1, 3, 3, T(=230)]
                origin_height_sequence = torch.from_numpy(np.concatenate(origin_height_out, axis=-1))               # [B=1, 1, T(=230)]
                origin_width_sequence = torch.from_numpy(np.concatenate(origin_width_out, axis=-1))                 # [B=1, 1, T(=230)]


                assert target_indices[-1] == len(img_out)-1
                assert len(target_indices) == self.batch_times
                target_indices = torch.FloatTensor(target_indices)[None, :]  # B,T

                yield image_sequence, timestamps, target_indices, first_time, \
                      rotation_vector_sequence, translation_vector_sequence, \
                      camera_nt_sequence, camera_depth_sequence, camera_K_sequence, camera_Kinv_sequence, \
                      origin_height_sequence, origin_width_sequence

                img_out = []
                times = []
                target_indices = []

                # homo_out = []
                rotation_vector_out = []
                translation_vector_out = []
                camera_nt_out = []
                camera_depth_out = []
                camera_K_out = []
                camera_Kinv_out = []
                origin_height_out = []
                origin_width_out = []

                first_time = False



# from metavision_core_ml.data.image_planar_motion_stream import PlanarMotionStream
from .my_image_planar_motion_stream import My_PlanarMotionStream
from metavision_core_ml.corner_detection.utils import get_harris_corners_from_image, project_points
class My_CornerPlanarMotionStream(My_PlanarMotionStream):
    """
    Generates a planar motion in front of the image, returning both images and Harris' corners

    Args:
        image_filename: path to image
        height: desired height
        width: desired width
        max_optical_flow_threshold_random_list
        max_frames: number of frames to stream
        rgb: color images or gray
        infinite: border is mirrored
        pause_probability: probability of stream to pause
        draw_corners_as_circle: if true corners will be 2 pixels circles
    """

    def __init__(self, image_filename, height, width, max_optical_flow_threshold_random_list,
                 max_frames=1000, rgb=False, infinite=True,
                 pause_probability=0.5, draw_corners_as_circle=True):

        max_optical_flow_threshold = np.random.choice(max_optical_flow_threshold_random_list, size=1)[0]


        super().__init__(image_filename, height, width, max_frames=max_frames, rgb=rgb, infinite=infinite,
                         pause_probability=pause_probability,
                         max_optical_flow_threshold=max_optical_flow_threshold
                        )
        self.iter = 0




    def __next__(self):
        if self.iter >= len(self.camera):
            raise StopIteration

        # G_0to2, ts = self.camera()
        G_0to2, ts, rotation_vector, translation_vector = self.camera.get_homography_ts_pose(self.height, self.width)   # get pose for each frame


        pose_dict = {
            'rotation_vector': rotation_vector,
            'translation_vector': translation_vector,
            'nt': self.camera.nt,
            'depth': self.camera.depth,
            'K': self.camera.K,
            'Kinv': self.camera.Kinv,

            'origin_height': float(self.frame_height),
            'origin_width': float(self.frame_width)
        }


        out = cv2.warpPerspective(
            self.frame,
            G_0to2,
            dsize=(self.frame_width, self.frame_height),
            borderMode=self.border_mode,
        )
        out = cv2.resize(out, (self.width, self.height), 0, 0, cv2.INTER_AREA)

        self.iter += 1

        ts *= self.dt

        return out, ts, pose_dict

