import argparse
import os
import numpy as np
import random
import math
import json

import cv2
import mmcv
import torch
import torch.distributed as dist
import PIL.Image
import PIL.ImageDraw
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils.general_utils import mkdir
from mmdet.models.detectors.condlanenet import CondLaneNet, CondLanePostProcessor

from tools.condlanenet.common import COLORS, parse_lanes


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='seg checkpoint file')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--hm_thr', type=float, default=0.5)
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--show_dst',
        default='./work_dirs/culane/watch',
        help='path to save visualized results.')
    parser.add_argument(
        '--result_dst',
        default='./work_dirs/culane/results',
        help='path to save results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def adjust_result(lanes, crop_bbox, img_shape, tgt_shape=(590, 1640)):

    def in_range(pt, img_shape):
        if pt[0] >= 0 and pt[0] < img_shape[1] and pt[1] >= 0 and pt[
                1] <= img_shape[0]:
            return True
        else:
            return False

    left, top, right, bot = crop_bbox
    h_img, w_img = img_shape[:2]
    crop_width = right - left
    crop_height = bot - top
    ratio_x = crop_width / w_img
    ratio_y = crop_height / h_img
    offset_x = (tgt_shape[1] - crop_width) / 2
    offset_y = top

    results = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            for pt in lanes[key]['points']:
                pt[0] = float(pt[0] * ratio_x + offset_x)
                pt[1] = float(pt[1] * ratio_y + offset_y)
                pts.append(pt)
            if len(pts) > 1:
                results.append(pts)
    return results


def adjust_point(hm_points,
                 downscale,
                 crop_bbox,
                 img_shape,
                 tgt_shape=(590, 1640)):
    left, top, right, bot = crop_bbox
    h_img, w_img = img_shape[:2]
    crop_width = right - left
    crop_height = bot - top
    ratio_x = crop_width / w_img
    ratio_y = crop_height / h_img
    offset_x = (tgt_shape[1] - crop_width) / 2
    offset_y = top
    coord_x = float((hm_points[0] + 0.5) * downscale * ratio_x + offset_x)
    coord_y = float((hm_points[1] + 0.5) * downscale * ratio_y + offset_y)
    coord_x = max(0, coord_x)
    coord_x = min(coord_x, tgt_shape[1])
    coord_y = max(0, coord_y)
    coord_y = min(coord_y, tgt_shape[0])
    return [coord_x, coord_y]


def out_result(lanes, dst=None):
    if dst is not None:
        with open(dst, 'w') as f:
            for lane in lanes:
                for idx, p in enumerate(lane):
                    if idx == len(lane) - 1:
                        print('{:.2f} '.format(p[0]), end='', file=f)
                        print('{:.2f}'.format(p[1]), file=f)
                    else:
                        print('{:.2f} '.format(p[0]), end='', file=f)
                        print('{:.2f} '.format(p[1]), end='', file=f)


def vis_one(results, filename, width=9, fps=None, num_lanes=None):
    """
    Visualize prediction and GT lanes and optionally overlay FPS + lane count.
    Here `fps` is interpreted as total FPS (inference + post-processing).
    """
    img = cv2.imread(filename)
    img_gt = cv2.imread(filename)
    img_pil = PIL.Image.fromarray(img)
    img_gt_pil = PIL.Image.fromarray(img_gt)
    num_failed = 0

    preds, annos = parse_lanes(results, filename, (590, 1640))

    for idx, anno_lane in enumerate(annos):
        PIL.ImageDraw.Draw(img_gt_pil).line(
            xy=anno_lane, fill=COLORS[idx + 1], width=width)
    for idx, pred_lane in enumerate(preds):
        PIL.ImageDraw.Draw(img_pil).line(
            xy=pred_lane, fill=COLORS[idx + 1], width=width)

    img = np.array(img_pil, dtype=np.uint8)
    img_gt = np.array(img_gt_pil, dtype=np.uint8)

    # Add FPS overlay (total FPS) to prediction image using OpenCV
    if fps is not None:
        # Color code based on FPS
        if fps < 2:
            fps_color = (0, 0, 255)      # Red - very slow
        elif fps < 4:
            fps_color = (0, 165, 255)    # Orange - slow
        elif fps < 6:
            fps_color = (0, 255, 255)    # Yellow - moderate
        else:
            fps_color = (0, 255, 0)      # Green - fast

        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(
            img,
            fps_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            fps_color,
            3
        )

        if num_lanes is not None:
            lane_text = f"Lanes: {num_lanes}"
            cv2.putText(
                img,
                lane_text,
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2
            )

    return img, img_gt, num_failed


def single_gpu_test(seg_model,
                    data_loader,
                    show=None,
                    hm_thr=0.3,
                    result_dst=None,
                    crop_bbox=[0, 270, 1640, 590],
                    nms_thr=4,
                    mask_size=(1, 40, 100)):
    seg_model.eval()
    dataset = data_loader.dataset
    post_processor = CondLanePostProcessor(
        mask_size=mask_size, hm_thr=hm_thr, use_offset=True, nms_thr=nms_thr)
    prog_bar = mmcv.ProgressBar(len(dataset))

    import time
    from collections import defaultdict

    inference_times = []
    total_times = []
    postproc_times = []

    # Track FPS and latencies by lane count
    inference_fps_by_lane_count = defaultdict(list)  # {num_lanes: [inference_fps1, ...]}
    total_fps_by_lane_count = defaultdict(list)      # {num_lanes: [total_fps1, ...]}
    inference_latency_by_lane_count = defaultdict(list)  # {num_lanes: [inference_time_ms1, ...]}
    total_latency_by_lane_count = defaultdict(list)      # {num_lanes: [total_time_ms1, ...]}
    postproc_latency_by_lane_count = defaultdict(list)   # {num_lanes: [postproc_time_ms1, ...]}

    for i, data in enumerate(data_loader):
        total_start = time.time()

        with torch.no_grad():
            sub_name = data['img_metas'].data[0][0]['sub_img_name']
            img_shape = data['img_metas'].data[0][0]['img_shape']
            sub_dst_name = sub_name.replace('.jpg', '.lines.txt')
            dst_dir = result_dst + sub_dst_name
            dst_folder = os.path.split(dst_dir)[0]
            mkdir(dst_folder)

            # Measure model inference time (forward pass only)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_start = time.time()
            seeds, hm = seg_model(
                return_loss=False, rescale=False, thr=hm_thr, **data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_times.append(inference_time)

            # Measure post-processing time
            postproc_start = time.time()
            downscale = data['img_metas'].data[0][0]['down_scale']
            lanes, seeds = post_processor(seeds, downscale)
            result = adjust_result(
                lanes=lanes,
                crop_bbox=crop_bbox,
                img_shape=img_shape,
                tgt_shape=(590, 1640))
            postproc_end = time.time()
            postproc_time = postproc_end - postproc_start
            postproc_times.append(postproc_time)

            out_result(result, dst=dst_dir)

            total_end = time.time()
            total_time = total_end - total_start
            total_times.append(total_time)

            # Calculate and display FPS
            current_inference_fps = 1.0 / inference_time if inference_time > 0 else 0
            current_total_fps = 1.0 / total_time if total_time > 0 else 0
            avg_inference_fps = (
                len(inference_times) / sum(inference_times)
                if sum(inference_times) > 0 else 0
            )

            # Count lanes for context
            num_lanes = len(lanes) if lanes else 0

            # Track FPS and latencies by lane count
            inference_fps_by_lane_count[num_lanes].append(current_inference_fps)
            total_fps_by_lane_count[num_lanes].append(current_total_fps)
            inference_latency_by_lane_count[num_lanes].append(
                inference_time * 1000)  # Forward-only (ms)
            total_latency_by_lane_count[num_lanes].append(
                total_time * 1000)  # Total (ms)
            # Define post-proc latency as (total - forward) in ms, as requested
            postproc_latency_by_lane_count[num_lanes].append(
                (total_time - inference_time) * 1000)

            # Display FPS info periodically
            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"\n[Frame {i+1}] "
                    f"Inference FPS: {current_inference_fps:.2f} | "
                    f"Avg Inference FPS: {avg_inference_fps:.2f} | "
                    f"Total FPS: {current_total_fps:.2f} | "
                    f"Lanes: {num_lanes} | "
                    f"Inference: {inference_time*1000:.1f}ms | "
                    f"Post-process: {postproc_time*1000:.1f}ms"
                )

        if show is not None and show:
            filename = data['img_metas'].data[0][0]['filename']
            # Use total FPS (inference + post-processing) for on-image display
            current_total_fps = 1.0 / total_time if total_time > 0 else 0
            num_lanes = len(lanes) if lanes else 0
            img_vis, img_gt_vis, num_failed = vis_one(
                result,
                filename,
                fps=current_total_fps,
                num_lanes=num_lanes
            )
            basename = '{}_'.format(num_failed) + sub_name[1:].replace('/', '.')
            dst_show_dir = os.path.join(show, basename)
            mkdir(show)
            cv2.imwrite(dst_show_dir, img_vis)
            dst_gt_dir = os.path.join(show, basename + '.gt.jpg')
            mkdir(show)
            cv2.imwrite(dst_gt_dir, img_gt_vis)

        batch_size = data['img'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    # Print final FPS statistics
    if inference_times:
        avg_inference_fps = len(inference_times) / sum(inference_times)
        min_inference_fps = (
            1.0 / max(inference_times) if max(inference_times) > 0 else 0
        )
        max_inference_fps = (
            1.0 / min(inference_times) if min(inference_times) > 0 else 0
        )
        avg_total_fps = (
            len(total_times) / sum(total_times) if sum(total_times) > 0 else 0
        )

        print(f"\n{'='*60}")
        print("FPS Statistics (Model Inference):")
        print(f"  Average Inference FPS: {avg_inference_fps:.2f}")
        print(f"  Min Inference FPS: {min_inference_fps:.2f} (slowest frame)")
        print(f"  Max Inference FPS: {max_inference_fps:.2f} (fastest frame)")
        print(f"  Average Total FPS (incl. post-processing): {avg_total_fps:.2f}")
        print(f"  Total frames processed: {len(inference_times)}")
        print(f"{'='*60}")

        # Average FPS and latencies by lane count
        all_lane_counts = set(inference_fps_by_lane_count.keys()) | set(
            total_fps_by_lane_count.keys())
        if all_lane_counts:
            print(f"\n{'='*110}")
            print("DETAILED PERFORMANCE METRICS BY LANE COUNT")
            print(f"{'='*110}")
            print(
                f"{'Lanes':<8} "
                f"{'Avg Inf FPS':<15} "
                f"{'Fwd Lat (ms)':<15} "
                f"{'Avg Tot FPS':<15} "
                f"{'Tot Lat (ms)':<15} "
                f"{'Post-Proc Lat (ms)':<24} "
                f"{'Count':<8}"
            )
            print(f"{'-'*110}")

            sorted_lane_counts = sorted(all_lane_counts)
            for num_lanes in sorted_lane_counts:
                # Inference FPS stats
                inf_fps_list = inference_fps_by_lane_count.get(num_lanes, [])
                if inf_fps_list:
                    avg_inf_fps = sum(inf_fps_list) / len(inf_fps_list)
                else:
                    avg_inf_fps = 0.0

                # Inference latency stats (ms)
                inf_lat_list = inference_latency_by_lane_count.get(
                    num_lanes, [])
                if inf_lat_list:
                    avg_inf_lat = sum(inf_lat_list) / len(inf_lat_list)
                else:
                    avg_inf_lat = 0.0

                # Total FPS stats
                tot_fps_list = total_fps_by_lane_count.get(num_lanes, [])
                if tot_fps_list:
                    avg_tot_fps = sum(tot_fps_list) / len(tot_fps_list)
                else:
                    avg_tot_fps = 0.0

                # Total latency stats (ms)
                tot_lat_list = total_latency_by_lane_count.get(
                    num_lanes, [])
                if tot_lat_list:
                    avg_tot_lat = sum(tot_lat_list) / len(tot_lat_list)
                else:
                    avg_tot_lat = 0.0

                # Post-processing latency stats (ms)
                post_lat_list = postproc_latency_by_lane_count.get(
                    num_lanes, [])
                if post_lat_list:
                    avg_post_lat = sum(post_lat_list) / len(post_lat_list)
                else:
                    avg_post_lat = 0.0

                count = max(len(inf_fps_list), len(tot_fps_list))
                lane_word = "lane" if num_lanes == 1 else "lanes"

                print(
                    f"{num_lanes} {lane_word:5s} "
                    f"{avg_inf_fps:>12.2f} "
                    f"{avg_inf_lat:>13.2f} "
                    f"{avg_tot_fps:>13.2f} "
                    f"{avg_tot_lat:>13.2f} "
                    f"{avg_post_lat:>22.2f} "
                    f"{count:>7}"
                )

            print(f"{'='*110}")
            print("Note: Fwd Lat = Forward pass latency (model inference only)")
            print("      Tot Lat = Total latency (inference + post-processing + I/O)")
            print("      Post-Proc Lat = Post-processing latency (lane extraction, NMS, coordinate transform)")
            print(f"{'='*110}")


class DateEnconding(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, np.float32):
            return float(o)


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])

    if not args.show:
        show_dst = None
    else:
        show_dst = args.show_dst
    if args.show is not None and args.show:
        mkdir(args.show_dst)

    single_gpu_test(
        model,
        data_loader,
        show=show_dst,
        hm_thr=args.hm_thr,
        result_dst=args.result_dst,
        crop_bbox=cfg.crop_bbox,
        nms_thr=cfg.nms_thr,
        mask_size=cfg.mask_size)


if __name__ == '__main__':
    main()
