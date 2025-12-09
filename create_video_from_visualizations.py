#!/usr/bin/env python3


import cv2
import numpy as np
import os
import glob
import re
from pathlib import Path
from collections import defaultdict

def natural_sort_key(text):
   
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]

def get_image_pairs(vis_dir):
  
    vis_dir = Path(vis_dir)
    
    if not vis_dir.exists():
        print(f"Error: Directory {vis_dir} does not exist!")
        return []
    
    # Get all .jpg files
    all_jpg_files = list(vis_dir.glob("*.jpg"))
    
    if not all_jpg_files:
        print(f"No .jpg files found in {vis_dir}")
        return []
    

    pred_images = []
    gt_images = {}
    
    for img_file in all_jpg_files:
        if img_file.name.endswith(".gt.jpg"):

            pred_name = img_file.name[:-7]  # Remove ".gt.jpg"
            gt_images[pred_name] = img_file
        else:

            pred_images.append(img_file)
    

    pairs = []
    for pred_img in sorted(pred_images, key=lambda x: natural_sort_key(x.name)):
        gt_img = gt_images.get(pred_img.name)
        if gt_img and gt_img.exists():
            pairs.append((pred_img, gt_img))
        else:

            pairs.append((pred_img, None))
    
    print(f"Found {len(pairs)} image pairs ({len([p for p in pairs if p[1] is not None])} with GT, {len([p for p in pairs if p[1] is None])} without GT)")
    return pairs


def create_video_with_fps_counter(image_pairs, output_path, fps=10, show_fps=True):

    if not image_pairs:
        print("No image pairs found!")
        return
    
    first_pred = cv2.imread(str(image_pairs[0][0]))
    first_gt = cv2.imread(str(image_pairs[0][1]))
    
    if first_pred is None or first_gt is None:
        print(f"Error: Could not read images from {image_pairs[0]}")
        return
    
    h, w = first_pred.shape[:2]
    
    frame_width = w * 2
    frame_height = h
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    
    print(f"Creating video: {output_path}")
    print(f"Resolution: {frame_width}x{frame_height}, Video FPS: {fps}")
    print(f"Total frames: {len(image_pairs)}")
    print("Note: Inference FPS and lane count are already on images from test script")
    
    for idx, (pred_path, gt_path) in enumerate(image_pairs):
        pred_img = cv2.imread(str(pred_path))
        if pred_img is None:
            print(f"Warning: Could not read {pred_path}")
            continue
        
        if gt_path is not None:
            gt_img = cv2.imread(str(gt_path))
            if gt_img is None:
                print(f"Warning: Could not read {gt_path}, using prediction only")
                gt_img = pred_img.copy()  # Use prediction as fallback
        else:
    
            gt_img = pred_img.copy()
        
 
        if pred_img.shape[:2] != (h, w):
            pred_img = cv2.resize(pred_img, (w, h))
        if gt_img.shape[:2] != (h, w):
            gt_img = cv2.resize(gt_img, (w, h))
        

        frame = np.hstack([pred_img, gt_img])
        

        out.write(frame)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_pairs)} frames...")
    
    out.release()
    print(f"\nVideo saved to: {output_path}")
    print(f"Total frames: {len(image_pairs)}")
    print("Video shows inference FPS and lane count from test script")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create video from visualization results')
    parser.add_argument('--input_dir', type=str, default='visualization_results',
                       help='Directory containing visualization images')
    parser.add_argument('--output', type=str, default='lane_detection_results.mp4',
                       help='Output video filename')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='Frames per second for the video')
    parser.add_argument('--no-fps-counter', action='store_true',
                       help='Disable FPS counter overlay')
    
    args = parser.parse_args()
    
    image_pairs = get_image_pairs(args.input_dir)
    
    if not image_pairs:
        print(f"No image pairs found in {args.input_dir}")
        print("Looking for pairs of: *.jpg and *.gt.jpg files")
        return
    
    print(f"Found {len(image_pairs)} image pairs")
    

    create_video_with_fps_counter(
        image_pairs, 
        args.output, 
        fps=args.fps,
        show_fps=not args.no_fps_counter
    )

if __name__ == '__main__':
    main()

