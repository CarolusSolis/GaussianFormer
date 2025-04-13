## Data

### NuScenes
In the data/nuscenes folder:

```bash
wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz
wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-test_meta.tgz
wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-test_blobs.tgz
wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval{01..10}_blobs.tgz
```

Uncompress all: 

```bash
for f in *.tgz; do tar -xvzf "$f"; done
```

### SurroundOcc
Go here to obtain download links: 
https://cloud.tsinghua.edu.cn/d/8dcb547238144d08a0bb/

Then download and unzip in data/surroundocc/

### pkl files:
Go here to obtain download links: 
https://cloud.tsinghua.edu.cn/d/bb96379a3e46442c8898/

Then download and unzip in data/nuscenes_cam/

## Model
Follow the README to download models under the ckpts/ directory.

Additionally, download the GaussianLifterV2 checkpoint as out/prob/init/init.pth:

```bash
wget https://cloud.tsinghua.edu.cn/seafhttp/files/2c138bf5-e2ab-4f87-96fe-6dd567f17f8a/lifter_10.pth -O out/prob/init/init.pth
```

## Docker Container setup
Follow the docker-README.md in the project root directory.

## Running the script

### Evaluation

```bash
python eval.py --py-config config/prob/nuscenes_gs25600.py --work-dir out/prob256/ --resume-from ckpts/Prob256_state_dict.pth     
```

### Training

```bash
python train.py --py-config config/prob/nuscenes_gs6400.py --work-dir out/prob6400/ --use-wandb --wandb-project gaussianformer2 --wandb-name test64
```
