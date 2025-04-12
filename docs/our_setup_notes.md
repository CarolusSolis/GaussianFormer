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

## Docker Container setup
Follow the docker-README.md in the project root directory.
