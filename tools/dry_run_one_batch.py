"""Dry-run: load config, build dataset/dataloader and time fetching one batch."""
import time
import sys
from mmcv import Config

try:
    from mmdet.datasets import build_dataset, build_dataloader
except Exception as e:
    print('Failed to import mmdet datasets:', e)
    sys.exit(1)

cfg = Config.fromfile('tools/openlane_small_train.py')

# Build dataset
print('Building dataset...')
start = time.time()
dataset = build_dataset(cfg.data.train)
print('Dataset built in %.3fs' % (time.time() - start))

# Build dataloader (single GPU, no dist)
print('Building dataloader...')
start = time.time()
dataloader = build_dataloader(
    dataset,
    samples_per_gpu=cfg.data['samples_per_gpu'],
    workers_per_gpu=cfg.data['workers_per_gpu'],
    num_gpus=1,
    dist=False,
    shuffle=False)
print('Dataloader built in %.3fs' % (time.time() - start))

# Fetch one batch and measure time
print('Fetching one batch (timing)...')
start = time.time()
try:
    batch = next(iter(dataloader))
    dt = time.time() - start
    print('Fetched one batch in %.3fs' % dt)
    # print batch keys and tensor shapes if possible
    if isinstance(batch, dict):
        for k, v in batch.items():
            try:
                print(k, type(v))
            except Exception:
                pass
except Exception as e:
    print('Error while fetching batch:', e)
    sys.exit(1)

print('Dry-run complete.')
