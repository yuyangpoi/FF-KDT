'''
last edit: 20240105
Yuyang
'''
import dv_processing as dv
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import glob
import argparse
import pytorch_lightning as pl
import csv

from event_utils.event_representation import event_EST
from eval.eval_tracker import EvalCornerTracker


from model_tracking.A_lightning_model_tracking_net_light_v2 import CornerDetectionLightningModel
from davis_utils.buffer import FrameBuffer, EventBuffer
from eval.buffer_manager import FeatureBuffer, TrajectoryBuffer, TimestampBuffer


torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


scene_frame_interval_dict = {
    'Normal': 25091,
    'Blur': 100000,
    'Dark': 25091,
    'Over': 40000
}


def parse_argument():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## dir params
    parser.add_argument('--file_dir', type=str,
                        help='The name of a aedat4 dir')
    parser.add_argument('--save_path', type=str, default='./result',
                        help='The save path for result .csv')

    parser.add_argument('--height', type=int, default=240, help='image height in evaluation')
    parser.add_argument('--width', type=int, default=320, help='image width in evaluation')

    parser.add_argument('--max_corner_num', type=int, default=100, help='max number of corners')
    parser.add_argument('--begin_frame', type=int, default=10, help='From which frame does the evaluation start')

    parser.add_argument('--show', default=True, help='if show detection and tracking process')


    ## detection model params
    parser.add_argument('--checkpoint', type=str,
                        default='../checkpoints/epoch=8-step=45000.ckpt',
                        help='checkpoint')
    parser.add_argument('--warping_event_volume_depth', type=int, default=10, help='event volume depth')
    parser.add_argument('--exposure_event_volume_depth', type=int, default=10, help='event volume depth')
    parser.add_argument('--needed_number_of_heatmaps', type=int, default=5,
                              help='number of target corner heatmaps')
    parser.add_argument('--num_tbins', type=int, default=5, help="timesteps per batch tbppt")
    parser.add_argument('--device', type=str, default='cuda:0', help="cpu or cuda:0")

    ## tracking model param
    parser.add_argument('--window_size', type=int, default=10, help="")
    parser.add_argument('--window_step', type=int, default=1, help="")


    return parser


def get_reader(file_path):
    assert os.path.exists(file_path), 'The file \'{}\' is not exist'.format(file_path)
    camera_reader = dv.io.MonoCameraRecording(file_path)

    return camera_reader



def init_model(params):
    params.warping_cin = params.warping_event_volume_depth * 2      # pos and neg
    params.exposure_cin = params.exposure_event_volume_depth * 2    # pos and neg
    params.cout = params.needed_number_of_heatmaps
    params.data_device = params.device
    params.height = params.height
    params.width = params.width

    model = CornerDetectionLightningModel(params, None, None)

    model = model.to(params.device)


    ckpt = params.checkpoint
    print('ckpt: ', ckpt)

    checkpoint = torch.load(ckpt, map_location=torch.device(params.device))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model




@torch.no_grad()
def preparation(lightning_model,
                frame,
                frame_interval_event_list,
                exposure_events,
                heatmap_T,
                ts, duration,

                feature_buffer,
                traj_buffer,
                timestamp_buffer):
    '''
    Args:
        lightning_model:
        frame: [B, 1, H, W]
        frame_interval_event_list: dshape [B, C, H, W]*4
        exposure_events: [B, C, H, W]
        heatmap_T: 5
        ts: accumulate time
        duration: frame interval
    Returns:
        features: [heatmap_T, B=1, C, Hc, Wc]
    '''
    lightning_model.eval()
    B = frame.shape[0]
    assert B == 1
    assert heatmap_T == len(frame_interval_event_list)
    delta_t = float(duration)

    ##
    ts_from_multi_time_steps = []
    interval_time = delta_t / (heatmap_T - 1)
    for sub_i in range(heatmap_T):
        ts_from_multi_time_steps.append(ts + sub_i * interval_time)

    ts_from_multi_time_steps = torch.tensor(ts_from_multi_time_steps)   # [heatmap_T]
    print('ts_from_multi_time_steps: ', ts_from_multi_time_steps)

    ##
    ts += delta_t


    ##
    pred_dict = lightning_model.detection_model(frame.float() / 255, exposure_events.float(), frame_interval_event_list)
    features = pred_dict['desc_list']  # list_len(10, [B, 128, H, W])

    features = torch.stack(features, dim=0)  # [heatmap_T, B=1, 128, Hc, Wc]


    ##
    feature_buffer.push(features)
    traj_buffer.push(None)
    timestamp_buffer.push(ts_from_multi_time_steps)

    return ts



@torch.no_grad()
def model_pred_and_save(lightning_model,
                        frame,
                        frame_interval_event_list,
                        exposure_events,
                        heatmap_T,

                        window_size,
                        window_step,

                        ts, duration, tracker,

                        feature_buffer,
                        traj_buffer,
                        timestamp_buffer,

                        first_time,

                        csv_writer=None,
                        show=False, frame_for_show=None):
    '''
    input:
        frame: [B=1, 1, H, W]
        frame_interval_event_list: dshape [B, C, H, W]*4
        exposure_events: [B, C, H, W]
    output:
        last_pred_features: [T, B=1, C=128, Hc, Wc]
        last_pred_traj: [T, B=1, 2, N]
    '''
    lightning_model.eval()
    B = frame.shape[0]
    assert B == 1
    assert window_step <= window_size // 2
    assert heatmap_T <= window_size
    assert heatmap_T == len(frame_interval_event_list)

    delta_t = float(duration)

    print('ts begin: ', ts)

    ##
    ts_from_multi_time_steps = []
    interval_time = delta_t / (heatmap_T - 1)
    for sub_i in range(heatmap_T):
        ts_from_multi_time_steps.append(ts + sub_i * interval_time)

    ts_from_multi_time_steps = torch.tensor(ts_from_multi_time_steps)   # [heatmap_T]
    print('ts_from_multi_time_steps: ', ts_from_multi_time_steps)

    ##
    ts += delta_t


    ##
    pred_dict = lightning_model.detection_model(frame.float() / 255, exposure_events.float(), frame_interval_event_list)
    heatmaps, features = pred_dict['heatmaps'], pred_dict['desc_list']   # [B, heatmap_T, H, W], list_len(10, [B, 64, H, W])

    features = torch.stack(features, dim=0)         # [T=10, B=1, 256, Hc, Wc]


    event_mask = torch.sum(torch.abs(torch.cat(frame_interval_event_list, dim=1)), dim=1, keepdim=True)
    event_mask = (event_mask!=0).float()    # [B=1, 1, H, W]
    current_corners, boundary_idexes, link_mask = tracker.add_and_link_corners(heatmap=heatmaps[:, 0, :, : ].unsqueeze(1) * event_mask,
                                                   new_threshold=0.95,
                                                   concat_threshold=0.5,
                                                   time_stamp=ts_from_multi_time_steps[0],
                                                   event_mask=None)
    current_corners = current_corners.to(lightning_model.device)



    ## 2.
    if current_corners.shape[0] > 0:

        if first_time:
            assert boundary_idexes == 0
            assert torch.all(link_mask)

            ##
            traj_buffer.buffer = current_corners.permute(1, 0).unsqueeze(0).unsqueeze(0).repeat(len(feature_buffer), 1, 1, 1)   # [T, B=1, 2, N]

        else:
            ##
            assert link_mask.shape[0] == traj_buffer.buffer.shape[-1]
            traj_buffer.buffer = traj_buffer.buffer[..., link_mask]

            ##
            if boundary_idexes < current_corners.shape[0]:
                new_corners = current_corners[boundary_idexes:]  # [N, 2]
                new_corners = new_corners.permute(1, 0).unsqueeze(0).unsqueeze(0).repeat(len(traj_buffer), 1, 1, 1) # [T, B, 2, N2]

                traj_buffer.buffer = torch.cat((traj_buffer.buffer, new_corners), dim=-1)   # [T, B, 2, N]



        ##
        ## 1.
        if boundary_idexes == 0:
            ##
            new_corners = current_corners[boundary_idexes:]     # [N, 2]
            query_idx = torch.zeros((B, current_corners.shape[0])).long().to(frame.device)  # [B=1, N]
            query_points = current_corners.permute(1, 0).unsqueeze(0)  # [B=1, 2, N]
            query_vector_for_tracker = lightning_model.tracking_model.extract_query_vectors(query_idx, query_points,
                                                                                            features)  # [B=1, C, N]

        ## 2.
        elif boundary_idexes == current_corners.shape[0]:
            query_vector_for_tracker = tracker.feature_vectors.permute(1, 0).unsqueeze(0)   # [B=1, C, N]

        ## 3.
        else:
            ##
            old_query_vector = tracker.feature_vectors.permute(1, 0).unsqueeze(0)           # [B=1, C, N]

            ##
            new_corners = current_corners[boundary_idexes:]                                 # [N2, 2]
            query_idx = torch.zeros((B, new_corners.shape[0])).long().to(frame.device)      # [B=1, N2]
            query_points = new_corners.permute(1, 0).unsqueeze(0)                           # [B=1, 2, N2]
            new_query_vector = lightning_model.tracking_model.extract_query_vectors(query_idx, query_points, features)  # [B=1, C, N2]

            query_vector_for_tracker = torch.cat((old_query_vector, new_query_vector), dim=-1)  # [B=1, C, N]



        assert heatmap_T % window_step == 0
        step_num = heatmap_T // window_step
        for i in range(step_num):
            print('\ni: ', i)
            print('i*window_step, (i+1)*window_step: ', i*window_step, (i+1)*window_step)

            feature_new = features[i*window_step:(i+1)*window_step]             # [window_step, B=1, 2, N]
            traj_new = traj_buffer.buffer[-1:].repeat(window_step, 1, 1, 1)     # [window_step, B=1, 2, N]
            ts_new = ts_from_multi_time_steps[i*window_step:(i+1)*window_step]  # [window_step]


            assert len(feature_buffer) == len(traj_buffer) == len(timestamp_buffer)
            old_window_len = window_size - window_step
            feature_old = feature_buffer.buffer[-old_window_len:]
            traj_old = traj_buffer.buffer[-old_window_len:]
            ts_old = timestamp_buffer.buffer[-old_window_len:]


            feature_for_tracker = torch.cat([feature_old, feature_new], dim=0).to(lightning_model.device)
            traj_for_tracker = torch.cat([traj_old, traj_new], dim=0).to(lightning_model.device)
            ts_for_tracker = torch.cat([ts_old, ts_new], dim=0).to(lightning_model.device)


            print('traj_for_tracker.shape: ', traj_for_tracker.shape)
            print('query_vector_for_tracker.shape: ', query_vector_for_tracker.shape)
            print('feature_for_tracker.shape: ', feature_for_tracker.shape)
            refine_traj = lightning_model.tracking_model(
                        traj_init=traj_for_tracker,
                        query_vectors_init=query_vector_for_tracker,
                        features=feature_for_tracker)[-1]   # [T, B, 2, N]

            ##
            feature_buffer.push(feature_new)
            traj_buffer.push(traj_new)
            traj_buffer.buffer[-window_size:] = refine_traj
            timestamp_buffer.push(ts_new)

            assert len(feature_buffer) == len(traj_buffer) == len(timestamp_buffer)


            ##
            update_corners = traj_buffer.buffer[-1]             # [B=1, 2, N]
            update_feature_vectors = query_vector_for_tracker   # [B=1, C, N]
            update_ts = timestamp_buffer.buffer[-1]             # float
            in_area_mask = tracker.update_corners(new_corners_loc=update_corners,
                                                  new_feature_vectors=update_feature_vectors,
                                                  time_stamp=update_ts,
                                                  image_size=frame.shape[-2:],
                                                  )

            ##
            traj_buffer.buffer = traj_buffer.buffer[..., in_area_mask]
            query_vector_for_tracker = query_vector_for_tracker[..., in_area_mask]


            ##
            tracker.save_nn_corners(tracker.current_corners, csv_writer, float(timestamp_buffer.buffer[-1]))



        first_time = False
        return first_time, ts




if __name__ == '__main__':
    ## Get params
    args, _ = parse_argument().parse_known_args(None)
    print(args)

    file_name_list = glob.glob(os.path.join(args.file_dir, '*.aedat4'))
    print(file_name_list)


    if args.file_dir[-1] == '/':
        args.file_dir = args.file_dir[:-1]
    scene_base_name = os.path.basename(args.file_dir)
    print('scene_base_name: ', scene_base_name)

    scene_frame_interval = scene_frame_interval_dict[scene_base_name]
    print('scene_frame_interval: ', scene_frame_interval)
    assert scene_frame_interval > 0


    save_dir = args.save_path + '_{}'.format(scene_base_name) + '_max{}'.format(args.max_corner_num)
    assert not os.path.exists(save_dir), '{}'.format(save_dir)
    os.makedirs(save_dir)


    for file_name in file_name_list:
        ## Create aedat4 reader
        reader = get_reader(file_name)
        camera_name = reader.getCameraName()
        raw_width, raw_height = reader.getFrameResolution()
        print('Camera name: {}'.format(camera_name))


        ## init model_detection
        lightning_model = init_model(args)
        device = lightning_model.device
        lightning_model.eval()



        ## Initialize a visualizer for the overlay
        visualizer = dv.visualization.EventVisualizer(reader.getEventResolution(),
                                                      dv.visualization.colors.white(),
                                                      dv.visualization.colors.blue(),
                                                      dv.visualization.colors.red())

        ## Create a window for image display
        if args.show:
            cv.namedWindow("Preview", cv.WINDOW_NORMAL)

        ## Create buffer for images and features
        buffer_max_len = args.window_size * 2
        frame_buffer = FrameBuffer(buffer_max_len)
        feature_buffer = FeatureBuffer(buffer_max_len)
        traj_buffer = TrajectoryBuffer(buffer_max_len)
        timestamp_buffer = TimestampBuffer(buffer_max_len)



        ## Create tracker
        tracker = EvalCornerTracker(max_corner_num=args.max_corner_num, device='cuda:0')
        ts_accumulated = 0  # timestamp accumulated from 0

        ## Create csv writer
        csv_save_path = os.path.join(save_dir, os.path.basename(file_name).split('.')[0]+'.csv')
        print(csv_save_path)
        csv_file = open(csv_save_path, 'w')
        csv_writer = csv.writer(csv_file)


        ## Continue the loop while both cameras are connected
        first_time = True
        while reader.isRunning():

            frame = reader.getNextFrame()
            if frame is not None:
                frame_buffer.push(frame)


                if len(frame_buffer) >= 2:

                    time_0 = time.time()

                    prev_frame = frame_buffer.get_prev_frame()
                    cur_frame = frame_buffer.get_cur_frame()

                    prev_exposure = prev_frame.exposure.microseconds
                    cur_exposure = cur_frame.exposure.microseconds

                    start_time, end_time = prev_frame.timestamp, cur_frame.timestamp
                    exposure_end_time = prev_frame.timestamp + prev_exposure
                    print('start_time, end_time: ', start_time, end_time)
                    print('exposure_end_time: ', exposure_end_time)
                    print('end_time - start_time: ', end_time - start_time)
                    print('exposure_end_time - start_time: ', exposure_end_time - start_time)

                    exposure_events = reader.getEventsTimeRange(start_time, exposure_end_time)
                    exposure_events_tensor, exposure_start_time, exposure_duration = EventBuffer.store_to_tensor(exposure_events, batch_idx=None, device=device) # [N, 4]
                    print('exposure_start_time, exposure_duration: ', exposure_start_time, exposure_duration)


                    ##
                    prev_frame_tensor = frame_buffer.frame_to_tensor(prev_frame, device=device) # [1, H, W]

                    exposure_EST = event_EST(exposure_events_tensor,
                                          raw_height, raw_width,
                                          torch.FloatTensor([exposure_start_time]).view(1, ).to(device),
                                          torch.FloatTensor([exposure_end_time-start_time]).view(1, ).to(device), # torch.FloatTensor([exposure_duration]).view(1, ).to(device),
                                          args.exposure_event_volume_depth,
                                          'bilinear')  # [C*2, H, W]


                    frame_interval_EST_list = []
                    time_interval = (end_time - start_time) / (lightning_model.hparams.cout - 1)
                    ## zero_event_EST
                    zero_event_start_time = start_time
                    zero_event_end_time = int(start_time + 0.005*time_interval)
                    print('zero_event_start_time, zero_event_end_time: ', zero_event_start_time, zero_event_end_time)
                    zero_events = reader.getEventsTimeRange(zero_event_start_time, zero_event_end_time)
                    zero_events_tensor, st_i, duration_i = EventBuffer.store_to_tensor(
                        zero_events,
                        batch_idx=None,
                        device=device)  # [N, 4]
                    print('end_time_i - start_time_i: ', zero_event_end_time - zero_event_start_time)
                    print('zero_st, zero_duration: ', st_i, duration_i)
                    zero_EST = event_EST(zero_events_tensor,
                                      raw_height, raw_width,
                                      torch.FloatTensor([st_i]).view(1, ).to(device),
                                      torch.FloatTensor([zero_event_end_time-zero_event_start_time]).view(1, ).to(device), # torch.FloatTensor([duration_i]).view(1, ).to(device),
                                      args.warping_event_volume_depth,
                                      'bilinear').float()  # [C*2, H, W]
                    frame_interval_EST_list.append(zero_EST)


                    ## warping event EST
                    for sub_i in range(lightning_model.hparams.cout-1):
                        # start_time_i = int(start_time + sub_i * time_interval)
                        # end_time_i = int(start_time + (sub_i+1) * time_interval)
                        start_time_i = start_time
                        end_time_i = int(start_time + (sub_i + 1) * time_interval)
                        print('sub_i, start_time_i, end_time_i: ', sub_i, start_time_i, end_time_i)

                        frame_interval_events_i = reader.getEventsTimeRange(start_time_i, end_time_i)
                        frame_interval_events_tensor_i, st_i, duration_i = EventBuffer.store_to_tensor(frame_interval_events_i,
                                                                                        batch_idx=None,
                                                                                        device=device)  # [N, 4]
                        print('end_time_i - start_time_i: ', end_time_i - start_time_i)
                        print('st_i, duration_i: ', st_i, duration_i)
                        EST_i = event_EST(frame_interval_events_tensor_i,
                                          raw_height, raw_width,
                                          torch.FloatTensor([st_i]).view(1, ).to(device),
                                          torch.FloatTensor([end_time_i-start_time_i]).view(1, ).to(device), # torch.FloatTensor([duration_i]).view(1, ).to(device),
                                          args.warping_event_volume_depth,
                                          'bilinear').float()  # [C*2, H, W]
                        frame_interval_EST_list.append(EST_i)





                    ##
                    prev_frame_tensor = F.interpolate(prev_frame_tensor.unsqueeze(0),
                                                 size=(args.height, args.width), mode='bilinear')    # [1, 1, H, W]
                    exposure_EST = F.interpolate(exposure_EST.unsqueeze(0),
                                                                size=(args.height, args.width), mode='bilinear')    # [1, C, H, W]


                    for i, frame_EST_i in enumerate(frame_interval_EST_list):
                        frame_interval_EST_list[i] = F.interpolate(frame_EST_i.unsqueeze(0),
                                                                size=(args.height, args.width), mode='bilinear')    # [1, C, H, W]



                    if len(frame_buffer) <= args.begin_frame:   #
                        ts_accumulated = preparation(lightning_model,
                                                    prev_frame_tensor,
                                                    frame_interval_EST_list,
                                                    exposure_EST,
                                                    args.needed_number_of_heatmaps,
                                                    ts_accumulated,
                                                    scene_frame_interval, # end_time - start_time,

                                                    feature_buffer,
                                                    traj_buffer,
                                                    timestamp_buffer,
                                                    )

                    else:   #
                        first_time, ts_accumulated = model_pred_and_save(lightning_model,
                                                  prev_frame_tensor,
                                                  frame_interval_EST_list,
                                                  exposure_EST,

                                                  args.needed_number_of_heatmaps,

                                                  args.window_size,
                                                  args.window_step,

                                                  ts_accumulated,
                                                  scene_frame_interval,     # end_time - start_time,

                                                  tracker,

                                                  feature_buffer,
                                                  traj_buffer,
                                                  timestamp_buffer,

                                                  first_time,

                                                  csv_writer,
                                                  show=False, frame_for_show=cur_frame.image)


                        assert len(feature_buffer) == len(traj_buffer) == len(timestamp_buffer)


                    ##
                    if args.show:
                        if tracker.current_corners.shape[0] > 0:
                            # cv.imshow('Preview', tracker.show(tracker.current_corners, cur_frame.image))
                            cv.imshow('Preview', tracker.show(tracker.current_corners,
                                                              cv.resize(cur_frame.image, dsize=(args.width, args.height), interpolation=cv.INTER_LINEAR)))
                        else:
                            # cv.imshow('Preview', cur_frame.image)
                            cv.imshow('Preview', cv.resize(cur_frame.image, dsize=(args.width, args.height), interpolation=cv.INTER_LINEAR))
                        # cv.waitKey(200)

                    # If escape button is pressed (code 27 is escape key), exit the program cleanly
                    if cv.waitKey(2) == 27:
                        exit(0)



        csv_file.close()


