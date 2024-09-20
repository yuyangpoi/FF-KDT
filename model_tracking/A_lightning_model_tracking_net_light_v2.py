import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os
import argparse
import numpy as np
import cv2
import random

from tqdm import tqdm
from itertools import islice

from metavision_core_ml.utils.show_or_write import ShowWrite
from model_detection.detection_net_v3_6 import FrameEventNet


from model_detection.detector_loss import fusion_consistensy_loss_multi_scale_N, peaky_loss, feature_alignment_loss, consistensy_loss_multi_scale_N
from model_tracking.tracking_net_light_v6_5 import FrameEventTrackingNet
from model_tracking.tracking_utils import get_topK_gt_trajectory, extract_query_vector, compute_correlation_map, get_topK_max_topk_indices
from model_tracking.tracker_loss import location_loss


class CornerDetectionCallback(pl.callbacks.Callback):
    """
    callbacks to our model_detection
    """

    def __init__(self, data_module, video_result_every_n_epochs=2):
        super().__init__()
        self.data_module = data_module
        self.video_every = int(video_result_every_n_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        torch.save(pl_module.detection_model,
                   os.path.join(trainer.log_dir, "whole-model_detection-epoch-{}.ckpt".format(trainer.current_epoch)))
        torch.save(pl_module.tracking_model,
                   os.path.join(trainer.log_dir, "whole-model_tracking-epoch-{}.ckpt".format(trainer.current_epoch)))
        if trainer.current_epoch==0 or not (trainer.current_epoch % self.video_every):
            pl_module.video(self.data_module.train_dataloader(), trainer.current_epoch, set="train")
            pl_module.video(self.data_module.val_dataloader(), trainer.current_epoch, set="val")




class CornerDetectionLightningModel(pl.LightningModule):
    """
    Corner Detection: Train your model_detection to predict corners as a heatmap
    """

    def __init__(self, hparams: argparse.Namespace,
                 detection_ckpt_path, tracking_ckpt_path) -> None:
        super().__init__()
        self.save_hyperparameters(hparams, logger=False)

        self.detection_model = FrameEventNet(frame_cin=1,
                                             exposure_event_cin=self.hparams.exposure_cin,
                                             warping_event_cin=self.hparams.warping_cin,
                                             cout=self.hparams.cout)

        self.tracking_model = FrameEventTrackingNet(feature_channels=self.detection_model.desc_channels,
                                                    scale_factor=self.detection_model.scale_factor,
                                                    patch_size=7,
                                                    iterations=4,
                                                    pyramid_layers=3)


        ## load checkpoint
        if detection_ckpt_path is not None:
            print('loading detection net checkpoint...')
            detection_checkpoint = torch.load(detection_ckpt_path, map_location=self.device)
            detection_ckpt_dict = {}
            for k, v in detection_checkpoint['state_dict'].items():
                detection_ckpt_dict[k.split('model.')[1]] = v
            print(detection_ckpt_dict.keys())
            self.detection_model.load_state_dict(detection_ckpt_dict)

        if tracking_ckpt_path is not None:
            print('loading tracking net checkpoint...')
            tracking_checkpoint = torch.load(tracking_ckpt_path, map_location=self.device)
            print(tracking_checkpoint.keys())
            self.tracking_model.load_state_dict(tracking_checkpoint)



        self.nan_cnt = 0
        self.few_event_cnt = 0



        ## Param in loss
        self.heatmap_consistensy_loss_alpha = 1.0
        self.alignment_loss_alpha = 10.0
        self.trajectory_loss_alpha = 1.0
        self.peaky_loss_alpha = 1.0
        self.warped_feature_loss_alpha = 1.0



        max_interval_in_loss = self.hparams.num_tbins * self.hparams.cout // 2
        self.interval_list_in_loss = [1, self.hparams.cout, max_interval_in_loss]  # [1, self.hparams.cout]
        self.N_in_loss_list = [20, 40]


        ##
        self.scale_factor = self.detection_model.scale_factor
        self.top_k = 128    # 100



        self.traj_window = 10

        if self.hparams.num_tbins == 5:
            self.traj_window_step_list = [1, 3, 5]
        else:
            raise NotImplementedError


        for i in range(len(self.traj_window_step_list)):
            assert self.traj_window_step_list[i] < self.traj_window
            assert ((self.hparams.num_tbins * self.hparams.cout) - self.traj_window) % self.traj_window_step_list[i] == 0


        self.peaky_loss_unmask_begin_epoch = 999999


        self.low_loss_cnt = 0




    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
        model = cls(hparams)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def compute_loss(self, frame,
                     frame_interval_events,
                     exposure_events,
                     reset_mask,
                     rotation_vectors, translation_vectors,
                     camera_nts, camera_depths,
                     camera_Ks, camera_Kinvs,
                     origin_sizes,
                     event_binary_masks,
                     frame_binary_masks,
                     clear_frame
                     ):


        loss_dict = {}
        self.detection_model.reset(reset_mask)
        self.detection_model.detach()


        warping_event_list_T = []
        for warping_t in range(frame_interval_events.shape[0]):
            warping_event_list = []
            for warping_i in range(self.hparams.cout):
                warping_event_list.append(frame_interval_events[warping_t, :, warping_i * self.hparams.warping_cin:(warping_i + 1) * self.hparams.warping_cin].float())
            warping_event_list_T.append(warping_event_list)


        heatmap_list = []
        mask_for_heatmap_list = []
        feature_list = []  #
        fusion_feature_list_list = []
        # warped_fusion_feature_list_list = []
        for i in range(frame_interval_events.shape[0]):
            output_dict = self.detection_model(frame[i].float() / 255, exposure_events[i].float(), warping_event_list_T[i])

            mask_for_heatmaps_i = event_binary_masks[i].float()  # (event_binary_masks[i] + frame_binary_masks[i]).float()

            mask_for_heatmap_list.append(mask_for_heatmaps_i)  # [B, 1, H, W]
            heatmap_list.append(output_dict['heatmaps'])  # [B, C=heatmap_T, H, W]

            feature_list.append(torch.stack(output_dict['desc_list'], dim=0))  # [heatmap_T=5, B, C=256, H, W]
            fusion_feature_list_list.append(output_dict['fusion_feature_list'])
            # warped_fusion_feature_list_list.append(output_dict['fusion_feature_warped_list'])

        mask_for_heatmaps = torch.stack(mask_for_heatmap_list, dim=0)   # [T, B, 1, H, W]
        pred_heatmaps = torch.stack(heatmap_list, dim=0)                # [T, B, C, H, W]
        features_all_BPTT_heatmapT = torch.cat(feature_list, dim=0)     # [BPTT_T*heatmap_T, B, C=256, H//8, W//8]




        ## 1. supervised detector
        ## 1.
        consistensy_loss_dict = consistensy_loss_multi_scale_N(pred_heatmaps*mask_for_heatmaps,
                                                               rotation_vectors, translation_vectors,
                                                               camera_nts, camera_depths,
                                                               camera_Ks, camera_Kinvs,
                                                               origin_sizes, (self.hparams.height, self.hparams.width),
                                                               N_list=self.N_in_loss_list,
                                                               pred_features=None,
                                                               interval_list=self.interval_list_in_loss)  # pred_features=features_all_BPTT_heatmapT)
        loss_dict["heatmap_consistensy_loss"] = self.heatmap_consistensy_loss_alpha * consistensy_loss_dict['heatmap_loss']
        if torch.isnan(loss_dict["heatmap_consistensy_loss"]):
            self.nan_cnt += 1  # TODO fix it
            loss_dict["heatmap_consistensy_loss"] = torch.zeros(1, requires_grad=True).to(self.device)

        ## 2.
        if self.current_epoch < self.peaky_loss_unmask_begin_epoch:
            loss_dict["peaky_loss"] = self.peaky_loss_alpha * peaky_loss(pred_heatmaps*mask_for_heatmaps,
                                                                    N=sorted(self.N_in_loss_list)[0],
                                                                    valid_mask=None)
        else:
            loss_dict["peaky_loss"] = self.peaky_loss_alpha * peaky_loss(pred_heatmaps,         # TODO: 和监督detection的时候不一样
                                                                    N=sorted(self.N_in_loss_list)[0],
                                                                    valid_mask=None)



        ## 2. supervise tracker
        BPTT_T, B, heatmap_T = pred_heatmaps.shape[:3]
        _, _, C, Hc, Wc = features_all_BPTT_heatmapT.shape

        window = self.traj_window  # heatmap_T*2
        window_step = random.choice(self.traj_window_step_list)  # heatmap_T
        slide_num = (BPTT_T * heatmap_T - window) // window_step + 1

        if window_step == 1:    # low memory
            top_k = self.top_k // 2
        else:
            top_k = self.top_k

        print('window, window_step, slide_num: ', window, window_step, slide_num)
        print('top_k: ', top_k)


        ## 2.1
        event_mask = event_binary_masks[0].float()   # torch.sum(events[0, :], dim=1).unsqueeze(1)  # [B, 1, H, W]
        border_mask = torch.zeros_like(event_mask)
        border_mask[:, :, 8:-8, 8:-8] = 1.0

        assert B == 1   # TODO:
        if torch.sum(event_mask).item() <= top_k * 10:
            mask_for_gt_trajectory = border_mask
            self.few_event_cnt += 1
        else:
            mask_for_gt_trajectory = event_mask * border_mask

        topK_gt_trajectory = get_topK_gt_trajectory(
            top_k,
            pred_heatmaps,
            0,
            rotation_vectors, translation_vectors,
            camera_nts, camera_depths,
            camera_Ks, camera_Kinvs,
            origin_sizes,
            (self.hparams.height, self.hparams.width),
            nms_kernel_size=sorted(self.N_in_loss_list)[0],
            event_mask=mask_for_gt_trajectory,
        )  # [T=BPTT_T*heatmap_T, B, 2, top_k]





        ## 2.2
        tracker_loss = 0
        for tt in range(slide_num):
            begin_idx_tt = tt * window_step
            end_idx_tt = begin_idx_tt + window




            print('tt: {}, begin_idx_tt: {}, end_idx_tt: {}'.format(tt, begin_idx_tt, end_idx_tt))


            if tt == 0:
                query_idx = torch.randint(0, window, size=(B, top_k)).to(frame.device)  # [B, top_k]
                query_point_tt = self.tracking_model.extract_query_points(topK_gt_trajectory[begin_idx_tt: end_idx_tt],
                                                                          query_idx)  # [B, 2, top_k]


                ##
                in_W = (query_point_tt[:, 0, :] >= 0) * (query_point_tt[:, 0, :] <= self.hparams.width - 1)
                in_H = (query_point_tt[:, 1, :] >= 0) * (query_point_tt[:, 1, :] <= self.hparams.height - 1)
                in_area_mask = (in_W * in_H)  # [B, top_k]
                assert in_area_mask.shape[0] == 1   # TODO: test
                in_area_mask = in_area_mask.squeeze(0)  # [top_k]
                print('torch.sum(in_area_mask): ', torch.sum(in_area_mask))

                while torch.sum(in_area_mask) == 0:  #
                    query_idx = torch.randint(0, window, size=(B, top_k)).to(frame.device)  # [B, top_k]
                    query_point_tt = self.tracking_model.extract_query_points(topK_gt_trajectory[begin_idx_tt: end_idx_tt], query_idx)  # [B, 2, top_k]
                    ##
                    in_W = (query_point_tt[:, 0, :] >= 0) * (query_point_tt[:, 0, :] <= self.hparams.width - 1)
                    in_H = (query_point_tt[:, 1, :] >= 0) * (query_point_tt[:, 1, :] <= self.hparams.height - 1)
                    in_area_mask = (in_W * in_H)  # [B, top_k]
                    assert in_area_mask.shape[0] == 1  # TODO: test
                    in_area_mask = in_area_mask.squeeze(0)  # [top_k]
                    print('torch.sum(in_area_mask): ', torch.sum(in_area_mask))
                    print('topK_gt_trajectory: ', topK_gt_trajectory)
                    print('query_idx: ', query_idx)
                    print('query_point_tt: ', query_point_tt)
                    print('torch.sum(event_mask): ', torch.sum(event_mask))
                    print('torch.sum(pred_heatmaps[0, :, 0]): ', torch.sum(pred_heatmaps[0, :, 0]))
                    print('torch.sum(event_mask) == 0: ', torch.sum(event_mask) == 0)
                    print('torch.sum(event_mask) < 10: ', torch.sum(event_mask) < 10)
                    print('torch.sum(event_mask).item() == 0: ', torch.sum(event_mask).item() == 0)
                    print('self.few_event_cnt: ', self.few_event_cnt)



                query_idx = query_idx[..., in_area_mask]
                query_point_tt = query_point_tt[..., in_area_mask]
                topK_gt_trajectory = topK_gt_trajectory[..., in_area_mask]






                topK_gt_trajectory_tt = topK_gt_trajectory[begin_idx_tt: end_idx_tt]  # [window, B, 2, top_k]
                features_tt = features_all_BPTT_heatmapT[begin_idx_tt: end_idx_tt]  # [window, B, C, Hc, Wc]
                # assert topK_gt_trajectory_tt.shape[0] == features_tt.shape[0] == self.window


                query_vectors_init = self.tracking_model.extract_query_vectors(query_idx, query_point_tt, features_tt)

                traj_init_tt = query_point_tt.unsqueeze(0).repeat(window, 1, 1, 1)


                pred_traj_list_tt = self.tracking_model(traj_init_tt,
                                                        query_vectors_init,
                                                        features_tt)  # list, dshape: [window, B, 2, top_k]

                tracker_loss += location_loss(pred_traj_list_tt, topK_gt_trajectory_tt,
                                              (self.hparams.height, self.hparams.width),
                                              query_idx=query_idx)

            else:
                topK_gt_trajectory_tt = topK_gt_trajectory[begin_idx_tt: end_idx_tt]  # [window, B, 2, top_k]
                features_tt = features_all_BPTT_heatmapT[begin_idx_tt: end_idx_tt]  # [window, B, C, Hc, Wc]


                first_harf_traj_init_tt = last_pred_traj    # [window-window_step, B, 2, top_k]
                second_harf_traj_init_tt = last_pred_traj[-1].unsqueeze(0).repeat(window_step, 1, 1, 1)  # [window_step, B, 2, top_k]
                traj_init_tt = torch.cat([first_harf_traj_init_tt, second_harf_traj_init_tt], dim=0)   # [window, B, 2, top_k]


                pred_traj_list_tt = self.tracking_model(traj_init_tt,  # last_pred_traj,   # TODO: 测试detach()
                                                        query_vectors_init,
                                                        features_tt)  # list, dshape: [window, B, 2, top_k]

                tracker_loss += location_loss(pred_traj_list_tt, topK_gt_trajectory_tt,
                                              (self.hparams.height, self.hparams.width),
                                              query_idx=None)

            last_pred_traj = pred_traj_list_tt[-1][-(window-window_step):]  # [window_step, B, 2, top_k]




        loss_dict['trajectory_loss'] = self.trajectory_loss_alpha * (tracker_loss / slide_num)
        if torch.isnan(loss_dict["trajectory_loss"]):
            self.nan_cnt += 1   # TODO fix it
            loss_dict["trajectory_loss"] = torch.zeros(1, requires_grad=True).to(self.device)



        return loss_dict


    def training_step(self, batch, batch_nb):
        # loss_dict = self.compute_loss(batch["blurred_images"], batch["events"], batch["corners"], batch["reset"], batch["homos"])
        loss_dict = self.compute_loss(batch["blurred_images"],
                                      batch["frame_interval_events"],
                                      batch["exposure_events"],
                                      batch["reset"],
                                      batch["rotation_vectors"], batch['translation_vectors'],
                                      batch["camera_nts"], batch['camera_depths'],
                                      batch["camera_Ks"], batch['camera_Kinvs'],
                                      batch['origin_sizes'],
                                      batch['event_binary_masks'],
                                      batch['frame_binary_masks'],
                                      batch['clear_images'],
                                      )
        loss = sum([v for k, v in loss_dict.items()])
        logs = {'loss': loss}
        logs.update({'train_' + k: v.item() for k, v in loss_dict.items()})


        for k, v in loss_dict.items():
            print('{}: {}'.format(k, v))
        print()

        learning_rate = self.optimizers().state_dict()['param_groups'][0]['lr']

        if loss_dict['trajectory_loss'] < 0.2:
            self.low_loss_cnt += 1


        self.log('learning_rate', learning_rate)
        self.log('nan_cnt', self.nan_cnt)
        self.log('few_event_cnt', self.few_event_cnt)
        self.log('low_loss_cnt', self.low_loss_cnt)

        print('lr: {}'.format(learning_rate))

        print('heatmap_consistensy_loss_alpha: {}'.format(self.heatmap_consistensy_loss_alpha))
        print('peaky_loss_alpha: {}'.format(self.peaky_loss_alpha))
        print('alignment_loss_alpha: {}'.format(self.alignment_loss_alpha))
        print('trajectory_loss_alpha: {}'.format(self.trajectory_loss_alpha))
        print('warped_feature_loss_alpha: ', self.warped_feature_loss_alpha)

        print('nan_cnt: {}'.format(self.nan_cnt))
        print('few_event_cnt: {}'.format(self.few_event_cnt))
        print('low_loss_cnt: {}'.format(self.low_loss_cnt))

        self.log('train_loss', loss)
        for k, v in loss_dict.items():
            self.log('train_' + k, v)

        return logs


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


    def make_heat_map_image(self, pred, divide_max=True):
        image = np.zeros((pred.shape[3], pred.shape[4]))
        for t in range(pred.shape[2]):
            pred_t = pred[0, 0, t]
            image = image + pred_t.cpu().numpy()
        if (image.max() != 0) and divide_max:
            image /= image.max()
        image *= 255
        image = np.concatenate([np.expand_dims(image, 2)] * 3, axis=2)
        return image.astype(np.uint8)

    def make_color_heat_map_image(self, pred, threshold=0.1):
        image = np.zeros((pred.shape[3], pred.shape[4], 3))
        pred = pred.cpu().numpy()
        for t in range(pred.shape[2]):
            pred_t = 1*(pred[0, 0, t] > threshold)
            image[pred_t != 0] = np.array([0, (pred.shape[2]-1-t)*(int(255/pred.shape[2])), 255])
        return image.astype(np.uint8)


    def image_from_events(self, events, mode='voxel_grid'):
        '''
        Args:
            events: [T, B, C, H, W]
            mode: string
        Returns:

        '''
        assert mode == 'voxel_grid' or mode == 'EST'
        if mode == 'voxel_grid':
            events = events.sum(2).unsqueeze(2)
        elif mode == 'EST':
            pos_events = events[:, :, 0::2]
            neg_events = events[:, :, 1::2]
            pos_events = pos_events.sum(2).unsqueeze(2)
            neg_events = neg_events.sum(2).unsqueeze(2)
            events = pos_events - neg_events

        events_as_image = 255 * (events > 0) + 0 * (events < 0) + 128 * (events == 0)
        return events_as_image


    def video(self, dataloader, epoch=0, set="val"):
        """

        Args:
            dataloader: data loader from train or val set
            epoch: epoch
            set: can be either train or val

        Returns:

        """
        print('Start Video on {} set **************************************************************'.format(set))

        self.detection_model.eval()
        self.tracking_model.eval()

        video_name = os.path.join(self.hparams.root_dir, 'videos', 'video_{}_{}.mp4'.format(set, epoch))
        dir = os.path.dirname(video_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        show_write = ShowWrite(False, video_name)

        with torch.no_grad():
            for batch in tqdm(islice(dataloader, self.hparams.demo_iter), total=self.hparams.demo_iter):
                frames = batch["blurred_images"].to(self.device)

                frame_interval_events = batch["frame_interval_events"].to(self.device)
                exposure_events = batch["exposure_events"].to(self.device)


                rotation_vectors = batch["rotation_vectors"].to(self.device)
                translation_vectors = batch['translation_vectors'].to(self.device)
                camera_nts = batch["camera_nts"].to(self.device)
                camera_depths = batch['camera_depths'].to(self.device)
                camera_Ks = batch["camera_Ks"].to(self.device)
                camera_Kinvs = batch['camera_Kinvs'].to(self.device)
                origin_sizes = batch['origin_sizes'].to(self.device)
                event_binary_mask = batch['event_binary_masks'].to(self.device)



                self.detection_model.reset(batch["reset"])



                warping_event_list_T = []
                for warping_t in range(frame_interval_events.shape[0]):
                    warping_event_list = []
                    for warping_i in range(self.hparams.cout):
                        warping_event_list.append(frame_interval_events[warping_t, :, warping_i * self.hparams.warping_cin:(warping_i + 1) * self.hparams.warping_cin].float())
                    warping_event_list_T.append(warping_event_list)



                feature_list = []
                pred_list = []
                warped_fusion_feature_list_list = []
                for i in range(frame_interval_events.shape[0]):
                    output_dict = self.detection_model(frames[i].float() / 255, exposure_events[i].float(), warping_event_list_T[i])
                    feature_list.append(torch.stack(output_dict['desc_list'], dim=0))
                    pred_list.append(output_dict['heatmaps'])
                    warped_fusion_feature_list_list.append(output_dict['fusion_feature_warped_list'])
                pred = torch.stack(pred_list, dim=0)    # [T, B, C=5, H, W]
                features_all_BPTT_heatmapT = torch.cat(feature_list, dim=0)    # [BPTT*heatmapT, B, C=256, H/8, W/8]


                query_idx = 0
                event_mask = event_binary_mask[query_idx].float() # torch.sum(events[0, :], dim=1).unsqueeze(1)  # [B, 1, H, W]
                border_mask = torch.zeros_like(event_mask)
                border_mask[:, :, 8:-8, 8:-8] = 1.0
                top1_gt_trajectory = get_topK_gt_trajectory(1, pred, query_idx,
                                                                 rotation_vectors, translation_vectors,
                                                                 camera_nts, camera_depths,
                                                                 camera_Ks, camera_Kinvs,
                                                                 origin_sizes,
                                                                 (self.hparams.height, self.hparams.width),
                                                                 nms_kernel_size=sorted(self.N_in_loss_list)[0],
                                                                 event_mask=event_mask*border_mask
                                                                 )  # [T=BPTT_T*heatmap_T, B, 2, top_k]

                query_vector = extract_query_vector(top1_gt_trajectory[query_idx, ..., 0]/self.scale_factor, features_all_BPTT_heatmapT[query_idx])
                correlation_map_begin = compute_correlation_map(query_vector, features_all_BPTT_heatmapT[0]) # [B, 1, H/8, W/8]
                correlation_map_end = compute_correlation_map(query_vector, features_all_BPTT_heatmapT[-1])  # [B, 1, H/8, W/8]
                correlation_map_begin_max_index = get_topK_max_topk_indices(correlation_map_begin, 1)        # ([B, 1], [B, 1])
                correlation_map_end_max_index = get_topK_max_topk_indices(correlation_map_end, 1)            # ([B, 1], [B, 1])

                ##
                correlation_map_begin_fow_show = (correlation_map_begin[0, 0].cpu().numpy()*255).astype(np.uint8)
                # correlation_map_begin_max_index = np.unravel_index(np.argmax(correlation_map_begin_fow_show, axis=None), correlation_map_begin_fow_show.shape)
                correlation_map_begin_fow_show = np.concatenate([np.expand_dims(correlation_map_begin_fow_show, 2)] * 3,
                                                                axis=2)
                correlation_map_begin_fow_show = cv2.resize(correlation_map_begin_fow_show, dsize=(self.hparams.width, self.hparams.height))
                cv2.circle(correlation_map_begin_fow_show, (int(correlation_map_begin_max_index[1][0].cpu().numpy()*self.scale_factor),
                                                            int(correlation_map_begin_max_index[0][0].cpu().numpy()*self.scale_factor)),
                           6, [0, 255, 0], thickness=1)
                cv2.circle(correlation_map_begin_fow_show, top1_gt_trajectory[0, 0, :, 0].cpu().numpy().astype(np.long),
                           8, [0, 0, 255], thickness=1)

                correlation_map_end_fow_show = (correlation_map_end[0, 0].cpu().numpy()*255).astype(np.uint8)
                # correlation_map_end_max_index = np.unravel_index(np.argmax(correlation_map_end_fow_show, axis=None), correlation_map_end_fow_show.shape)
                correlation_map_end_fow_show = np.concatenate([np.expand_dims(correlation_map_end_fow_show, 2)] * 3,
                                                              axis=2)
                correlation_map_end_fow_show = cv2.resize(correlation_map_end_fow_show, dsize=(self.hparams.width, self.hparams.height))
                cv2.circle(correlation_map_end_fow_show, (int(correlation_map_end_max_index[1][0].cpu().numpy()*self.scale_factor),
                                                          int(correlation_map_end_max_index[0][0].cpu().numpy()*self.scale_factor)), 6,
                           [0, 255, 0], thickness=1)
                cv2.circle(correlation_map_end_fow_show, top1_gt_trajectory[-1, 0, :, 0].cpu().numpy().astype(np.long),
                           8, [0, 0, 255], thickness=1)



                topK_gt_trajectory = get_topK_gt_trajectory(self.top_k, pred, query_idx,
                                                            rotation_vectors, translation_vectors,
                                                            camera_nts, camera_depths,
                                                            camera_Ks, camera_Kinvs,
                                                            origin_sizes,
                                                            (self.hparams.height, self.hparams.width),
                                                            nms_kernel_size=sorted(self.N_in_loss_list)[0],
                                                            event_mask=event_mask * border_mask
                                                            )  # [T=BPTT_T*heatmap_T, B, 2, top_k]
                BPTT_T, B, heatmap_T = pred.shape[:3]
                _, _, C, Hc, Wc = features_all_BPTT_heatmapT.shape
                # print(BPTT_T, B, heatmap_T, C, Hc, Wc)
                # print('topK_gt_trajectory.shape：', topK_gt_trajectory.shape)
                topK_gt_trajectory = topK_gt_trajectory.reshape(BPTT_T, heatmap_T,
                                                                B, 2, self.top_k)  # [BPTT_T, heatmap_T, B, 2, top_k]
                features_all_BPTT_heatmapT = features_all_BPTT_heatmapT.reshape(BPTT_T, heatmap_T, B, C, Hc, Wc)






                ##
                warped_feature_heatmap_list = []
                for warped_fusion_feature_list in warped_fusion_feature_list_list:
                    waped_feature_heatmap_c_list = []
                    for warped_fusion_feature in warped_fusion_feature_list:
                        __warped_fusion_feature = self.detection_model.final_conv(warped_fusion_feature)
                        waped_feature_heatmap = self.detection_model.heatmap_head(__warped_fusion_feature)  # [B, 1, H, W]
                        waped_feature_heatmap_c_list.append(waped_feature_heatmap)
                    warped_feature_heatmap_list.append(torch.cat(waped_feature_heatmap_c_list, dim=1))  # [B, heatmap_T, H, W]
                warped_feature_heatmaps = torch.stack(warped_feature_heatmap_list, dim=0)  # [BPTT_T, B, heatmap_T, H, W]







                image = self.image_from_events(frame_interval_events, mode='EST')
                val_tracker_loss = 0
                val_loss_dict = {}
                for t in range(pred.shape[0]):

                    ## observe val loss
                    topK_gt_trajectory_tt = topK_gt_trajectory[t]  # [heatmap_T, B, 2, top_k]
                    if t == 0:
                        query_point_tt = topK_gt_trajectory_tt[query_idx]  # [B, 2, top_k]
                    else:
                        query_point_tt = topK_gt_trajectory[t - 1][-1]  #
                    features_tt = features_all_BPTT_heatmapT[t]  # [heatmap_T, B, C, Hc, Wc]


                    query_idx_batch = torch.tensor(query_idx).unsqueeze(0).unsqueeze(0).repeat(B, topK_gt_trajectory.shape[-1]).to(query_point_tt.device) # [B, N]
                    query_vectors_init_tt = self.tracking_model.extract_query_vectors(query_idx_batch, query_point_tt, features_tt)

                    traj_init_tt = query_point_tt.unsqueeze(0).repeat(heatmap_T, 1, 1, 1)
                    pred_traj_list_tt = self.tracking_model(
                        traj_init=traj_init_tt,
                        query_vectors_init=query_vectors_init_tt,
                        features=features_tt)  # list, dshape: [heatmap_T, B, 2, top_k]

                    val_tracker_loss += location_loss(pred_traj_list_tt, topK_gt_trajectory_tt,
                                                      (self.hparams.height, self.hparams.width),
                                                      query_idx_batch)


                    pred_t = pred[t, 0].unsqueeze(0).unsqueeze(1)
                    heat_map_image = self.make_color_heat_map_image(pred_t, threshold=0.8)

                    heatmaps = (pred_t[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    heatmap_list = []
                    for i in range(self.hparams.cout):
                        heatmap_list.append(np.stack([heatmaps[i]] * 3, axis=-1))
                    # print('last_heatmap.shape: ', last_heatmap.shape)



                    ##
                    warped_feature_heatmap_t = warped_feature_heatmaps[t, 0].unsqueeze(0).unsqueeze(1)

                    warped_feature_heatmaps_t = (warped_feature_heatmap_t[0, 0].cpu().numpy() * 255).astype(
                        np.uint8)
                    warped_feature_heatmaps_t_list = []
                    for i in range(self.hparams.cout):
                        warped_feature_heatmaps_t_list.append(np.stack([warped_feature_heatmaps_t[i]] * 3, axis=-1))




                    events_image = image[t, 0, 0].cpu().numpy().astype(np.uint8)
                    events_image = np.concatenate([np.expand_dims(events_image, 2)] * 3, axis=2)

                    blurred_image = frames[t, 0, 0].cpu().numpy().astype(np.uint8)
                    blurred_image = np.concatenate([np.expand_dims(blurred_image, 2)] * 3, axis=2)


                    ##
                    topK_gt_trajectory_tt = topK_gt_trajectory[t]  # [heatmap_T, B, 2, top_k]
                    if t == 0:
                        query_point_tt = topK_gt_trajectory_tt[query_idx]  # [B, 2, top_k]
                    else:
                        query_point_tt = topK_gt_trajectory[t - 1][-1]  #
                    features_tt = features_all_BPTT_heatmapT[t]  # [heatmap_T, B, C, Hc, Wc]

                    query_idx_batch = torch.tensor(query_idx).unsqueeze(0).unsqueeze(0).repeat(B, topK_gt_trajectory.shape[-1]).to(query_point_tt.device)  # [B, N]
                    query_vectors_init_tt = self.tracking_model.extract_query_vectors(query_idx_batch, query_point_tt, features_tt)

                    traj_init_tt = query_point_tt.unsqueeze(0).repeat(heatmap_T, 1, 1, 1)
                    pred_traj_list_tt = self.tracking_model(
                        traj_init=traj_init_tt,
                        query_vectors_init=query_vectors_init_tt,
                        features=features_tt)  # list, dshape: [heatmap_T, B, 2, top_k]

                    gt_trajectory = topK_gt_trajectory_tt.cpu().numpy().astype(np.int)  # [T=5, B, 2, top_k]
                    pred_trajectory_list = [pred_traj_list_tt[-3].cpu().numpy().astype(np.int),
                                            pred_traj_list_tt[-2].cpu().numpy().astype(np.int),
                                            pred_traj_list_tt[-1].cpu().numpy().astype(np.int)]  # [T=5, B, 2, top_k]
                    gt_trajectory_fow_show = np.copy(events_image)
                    pred_trajectory_fow_show_list = [np.copy(events_image), np.copy(events_image), np.copy(events_image)]
                    for kk in range(self.top_k):
                        color_index = random.randint(0, 2)
                        for tt in range(self.hparams.cout):
                            color_tt = [0, 0, 0]
                            color_tt[color_index] = int((255 / 4) * tt)

                            # print('gt_trajectory[tt, 0, :, kk]: ', gt_trajectory[tt, 0, :, kk])
                            # print('gt_trajectory[tt, 0, :, kk].shape: ', gt_trajectory[tt, 0, :, kk].shape)
                            # print('gt_trajectory[tt, 0, :, kk].dtype: ', gt_trajectory[tt, 0, :, kk].dtype)

                            if gt_trajectory[tt, 0, :, kk][0] >= 0 and \
                                gt_trajectory[tt, 0, :, kk][1] >= 0 and \
                                gt_trajectory[tt, 0, :, kk][0] < self.hparams.width and \
                                gt_trajectory[tt, 0, :, kk][1] < self.hparams.height:
                                gt_trajectory_fow_show = cv2.circle(gt_trajectory_fow_show,
                                                                    gt_trajectory[tt, 0, :, kk],
                                                                    2, color_tt, thickness=2)

                            for iii in range(3):

                                # print('iii: ', iii)
                                #
                                # print('gt_trajectory[tt, 0, :, kk]: ', gt_trajectory[tt, 0, :, kk])
                                # print('gt_trajectory[tt, 0, :, kk].shape: ', gt_trajectory[tt, 0, :, kk].shape)
                                # print('gt_trajectory[tt, 0, :, kk].dtype: ', gt_trajectory[tt, 0, :, kk].dtype)
                                #
                                # print('pred_trajectory_list[iii][tt, 0, :, kk]: ', pred_trajectory_list[iii][tt, 0, :, kk])
                                # print('pred_trajectory_list[iii].shape: ', pred_trajectory_list[iii].shape)
                                # print('pred_trajectory_list[iii].dtype: ', pred_trajectory_list[iii].dtype)


                                if pred_trajectory_list[iii][tt, 0, :, kk][0] >= 0 and \
                                        pred_trajectory_list[iii][tt, 0, :, kk][1] >= 0 and \
                                        pred_trajectory_list[iii][tt, 0, :, kk][0] < self.hparams.width and \
                                        pred_trajectory_list[iii][tt, 0, :, kk][1] < self.hparams.height:
                                    pred_trajectory_fow_show_list[iii] = cv2.circle(pred_trajectory_fow_show_list[iii],
                                                                                    pred_trajectory_list[iii][tt, 0, :, kk],
                                                                                    2, color_tt, thickness=2)





                    ## 
                    top1_gt_trajectory_t = top1_gt_trajectory[t*self.hparams.cout:(t+1)*self.hparams.cout]    # [5, B, 2, top_k]
                    query_vector_t = extract_query_vector(top1_gt_trajectory_t[0, ..., 0]/self.scale_factor, feature_list[t][0])
                    correlation_map_list = []
                    for i in range(self.hparams.cout):
                        correlation_map_t_i = compute_correlation_map(query_vector_t, feature_list[t][i])  # [B, 1, H, W]
                        correlation_map_max_index_t = get_topK_max_topk_indices(correlation_map_t_i, 1)  # ([B, 1], [B, 1])
                        correlation_map_t_i_fow_show = (correlation_map_t_i[0, 0].cpu().numpy()*255).astype(np.uint8)
                        correlation_map_t_i_fow_show = np.concatenate([np.expand_dims(correlation_map_t_i_fow_show, 2)] * 3, axis=2)
                        correlation_map_t_i_fow_show = cv2.resize(correlation_map_t_i_fow_show, dsize=(self.hparams.width, self.hparams.height))
                        cv2.circle(correlation_map_t_i_fow_show,
                                   (int(correlation_map_max_index_t[1][0].cpu().numpy()*self.scale_factor),
                                    int(correlation_map_max_index_t[0][0].cpu().numpy()*self.scale_factor)),
                                   6, [0, 255, 0], thickness=1)

                        if top1_gt_trajectory_t[i, 0, :, 0][0] >= 0 and \
                                top1_gt_trajectory_t[i, 0, :, 0][1] >= 0 and \
                                top1_gt_trajectory_t[i, 0, :, 0][0] < self.hparams.width and \
                                top1_gt_trajectory_t[i, 0, :, 0][1] < self.hparams.height:
                            cv2.circle(correlation_map_t_i_fow_show,
                                       top1_gt_trajectory_t[i, 0, :, 0].cpu().numpy().astype(np.long),
                                       8, [0, 0, 255], thickness=1)
                        correlation_map_list.append(correlation_map_t_i_fow_show)


                    ## 绘制mask
                    event_binary_mask_for_show = event_binary_mask[t, 0].squeeze().float().cpu().numpy()
                    event_binary_mask_for_show = (cv2.merge([event_binary_mask_for_show,
                                                             event_binary_mask_for_show,
                                                             event_binary_mask_for_show]) * 255).astype(np.uint8)



                    if True:
                        heat_map_image_mask = heat_map_image.sum(2) == 0
                        heat_map_image[heat_map_image_mask] = events_image[heat_map_image_mask]
                        heat_map_image[~heat_map_image_mask] = heat_map_image[~heat_map_image_mask]

                    # cat_0 = np.concatenate([blurred_image, events_image, heat_map_image, correlation_map_begin_fow_show, correlation_map_end_fow_show], axis=1)  # [180, 240*4, 3]
                    cat_0 = np.concatenate([blurred_image, events_image, heat_map_image, event_binary_mask_for_show, correlation_map_end_fow_show], axis=1)  # [180, 240*4, 3]
                    cat_1 = np.concatenate([heatmap_list[0], heatmap_list[1], heatmap_list[2], heatmap_list[3], heatmap_list[4]], axis=1)  # [180, 240*4, 3]
                    cat_3 = np.concatenate([correlation_map_list[0], correlation_map_list[1], correlation_map_list[2],
                                            correlation_map_list[3], correlation_map_list[4]], axis=1)  # [180, 240*4, 3]
                    cat_5 = np.concatenate([events_image, gt_trajectory_fow_show,
                                            pred_trajectory_fow_show_list[-3],
                                            pred_trajectory_fow_show_list[-2],
                                            pred_trajectory_fow_show_list[-1]], axis=1)  # [180, 240*4, 3]
                    cat_6 = np.concatenate([warped_feature_heatmaps_t_list[0], warped_feature_heatmaps_t_list[1],
                                            warped_feature_heatmaps_t_list[2], warped_feature_heatmaps_t_list[3],
                                            warped_feature_heatmaps_t_list[4]], axis=1)  # [180, 240*4, 3]
                    cat = np.concatenate([cat_0, cat_1, cat_3, cat_5, cat_6], axis=0)  # [180*3, 240*5, 3]
                    # event_image is an image created from events
                    # heatmap gt is the ground truth heatmap of corners overlaid with the events
                    # heatmap image is the predicted corners overlaid with the events
                    show_write(cat)


                val_loss_dict['val_trajectory_loss'] = self.trajectory_loss_alpha * (val_tracker_loss / BPTT_T)
                for k, v in val_loss_dict.items():
                    print('{}: {}'.format(k, v))
                    self.log(k, v)




        self.detection_model.train()
        self.tracking_model.train()
        print('Finish Video on {} set **************************************************************'.format(set))
