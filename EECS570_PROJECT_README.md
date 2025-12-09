## Setup environment:

1.	Load Modules
module purge
module load python/3.9.12
module load cuda/10.2.89
module load gcc/8.2.0

2. Install requirements
pip install -r requirements.txt

3. Install torch
python -m pip install --user "torch==1.10.0+cu102" "torchvision==0.11.1+cu102" --index-url https://download.pytorch.org/whl/cu102

4. Install mmcv
python -m pip install --user "mmcv-full==1.4.8" -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html

If you get issues later install this :
python -m pip uninstall -y mmcv-full
python -m pip install --user "mmcv-full==1.3.17" -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html

5. Load Setup
python setup.py clean
python setup.py develop --user

6.Might need to change
pip uninstall yapf -y
pip install yapf==0.32.0

7. Request GPU
salloc --account=eecs570f25s001_class --partition=gpu --gres=gpu:1 --time=2:00:00 --mem=64G


## Downloading dataset 

link :https://drive.google.com/file/d/1u7ck1c7fPNsZWocIg_S7RdXTUngg2LS8/view

1. Generate access token: 
- Go to OAuth 2.0 Playground https://developers.google.com/oauthplayground/
- In the Select the Scope box, paste https://www.googleapis.com/auth/drive.readonly
- Click Authorize APIs and then Exchange authorization code for tokens
- Copy the ACCESS_TOKEN

2. In terminal:
cd images
curl -H "Authorization: Bearer ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1u7ck1c7fPNsZWocIg_S7RdXTUngg2LS8?alt=media -o images.tar
tar -xvf images.tar --ignore-zeros
cd ..


## Training new model:

1. Converts JSON files to .lines.txt format for training/validation
python maketxt.py

2. Converts validation data (currently configured for validation only)
python convert_labels_final.py

3. Start training
CUDA_VISIBLE_DEVICES=0 PORT=29500 tools/dist_train.sh tools/openlane_small_train.py 1

## Testing new model:

python tools/condlanenet/culane/test_culane.py \
    tools/openlane_small_test.py \
    work_dirs/openlane/small/epoch_2.pth \
    --show \
    --show_dst visualization_results

## Getting GFLOP value:

python tools/get_flops.py configs/openlane_small_test.py --shape 800 544

##Getting sprint FPS plot test:

python sprint_plot.py

##Getting FPS plot after test:

python plot.py
