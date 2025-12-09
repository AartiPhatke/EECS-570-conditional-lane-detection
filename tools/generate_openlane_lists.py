#!/usr/bin/env python3

import argparse
from pathlib import Path

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def gather_images(src_dir: Path):
   
    if not src_dir.exists():
        return []
    imgs = [p for p in src_dir.rglob('*') if p.suffix.lower() in EXTS]
    imgs_sorted = sorted(imgs)
    return imgs_sorted


def write_list(paths, out_file: Path, root: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open('w') as f:
        for p in paths:
            # write relative path from root
            rel = p.relative_to(root)
            f.write(str(rel).replace('\\', '/') + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='Dataset root (default: .)')
    ap.add_argument('--train-dir', default='images/images/training', help='relative path to training images')
    ap.add_argument('--val-dir', default='images/images/validation', help='relative path to validation images')
    ap.add_argument('--test-list', default=None, help='optional: path to existing test list to copy to images/list/test.txt')
    ap.add_argument('--lane3d-dir', default=None, help='optional: path to lane3d folder (e.g. images/lane3d_300) to use its lists instead of scanning all images')
    ap.add_argument('--test-from-lane3d', action='store_true', help='when using --lane3d-dir, build test list from lane3d/test/*.txt (all files concatenated)')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    train_dir = root / args.train_dir
    val_dir = root / args.val_dir

    train_imgs = gather_images(train_dir)
    val_imgs = gather_images(val_dir)

    list_dir = root / 'images' / 'list'
    train_list = list_dir / 'train.txt'
    val_list = list_dir / 'valid.txt'
    test_list = list_dir / 'test.txt'

   
    if args.lane3d_dir:
        lane3d_root = (root / args.lane3d_dir).resolve()
        lane_train = lane3d_root / 'training' / 'training_list.txt'
        lane_val = lane3d_root / 'validation' / 'validation_list.txt'

        def read_lane_list(p: Path):
            if not p.exists():
                return []
            lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
            return lines

        lane_train_list = read_lane_list(lane_train)
        lane_val_list = read_lane_list(lane_val)

     
        mapped_train = []
        for rel in lane_train_list:
            img_rel = Path(args.train_dir) / Path(rel).with_suffix('.jpg')
            img_abs = root / img_rel
            if img_abs.exists():
                mapped_train.append(img_abs)
            else:
                alt = img_abs.with_suffix('.jpeg')
                if alt.exists():
                    mapped_train.append(alt)

        mapped_val = []
        for rel in lane_val_list:
            img_rel = Path(args.val_dir) / Path(rel).with_suffix('.jpg')
            img_abs = root / img_rel
            if img_abs.exists():
                mapped_val.append(img_abs)
            else:
                alt = img_abs.with_suffix('.jpeg')
                if alt.exists():
                    mapped_val.append(alt)

        print(f'Lane3D lists found: train {len(mapped_train)} images, val {len(mapped_val)} images')
        write_list(mapped_train, train_list, root)
        write_list(mapped_val, val_list, root)


        if args.test_from_lane3d:
            test_dir = lane3d_root / 'test'
            test_files = []
            if test_dir.exists():
                for tf in sorted(test_dir.glob('*.txt')):
                    test_files.extend(read_lane_list(tf))
            mapped_test = []
            for rel in test_files:
                img_rel = Path(args.val_dir) / Path(rel).with_suffix('.jpg')
                img_abs = root / img_rel
                if img_abs.exists():
                    mapped_test.append(img_abs)
            write_list(mapped_test, test_list, root)
        elif args.test_list:
            src = Path(args.test_list)
            if src.exists():
                with src.open('r') as fr, test_list.open('w') as fw:
                    for line in fr:
                        fw.write(line)
                print(f'Copied test list from {src} to {test_list}')
        else:
            write_list(mapped_val, test_list, root)

        print('Wrote:')
        print('  ', train_list)
        print('  ', val_list)
        print('  ', test_list)
        return

    print(f'Found {len(train_imgs)} training images, {len(val_imgs)} validation images')

    write_list(train_imgs, train_list, root)
    write_list(val_imgs, val_list, root)

    if args.test_list:
        src = Path(args.test_list)
        if src.exists():
            with src.open('r') as fr, test_list.open('w') as fw:
                for line in fr:
                    fw.write(line)
            print(f'Copied test list from {src} to {test_list}')
    else:
        write_list(val_imgs, test_list, root)

    print('Wrote:')
    print('  ', train_list)
    print('  ', val_list)
    print('  ', test_list)


if __name__ == '__main__':
    main()
