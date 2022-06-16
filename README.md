# Sylph: A Hypernetwork Framework for Incremental Few-shot Object Detection
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Authors: [Li Yin](https://github.com/liyin2015), [Juan M Perez-Rua](https://github.com/jperezrua), [Kevin J Liang](https://github.com/kevinjliang)


This repository is the official PyTorch implementation of [Sylph: A Hypernetwork Framework for Incremental Few-shot Object Detection](https://arxiv.org/abs/2203.13903), accepted to [CVPR 2022](https://cvpr2022.thecvf.com/).

### Citation
If you find any part of our paper or this codebase useful, please consider citing our paper:

```
@inproceedings{yin2022sylph,
  title={Sylph: A Hypernetwork Framework for Incremental Few-shot Object Detection},
  author={Yin, Li and Perez-Rua, Juan M and Liang, Kevin J},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9035--9045},
  year={2022}
}
```

### License
Please see [LICENSE.md](https://github.com/facebookresearch/sylph-few-shot-detection/blob/main/LICENSE.md) for more details.

## Install packages
Requires Python 3.8+.

Install virtualenv if not installed yet:
```
python3 -m pip install --user virtualenv
```

Create an environment:
```
python3 -m venv env
```

Activate the environment:
```
source env/bin/activate
```

Install dependencies:
```
pip install torch
pip3 install -r requirements.txt --find-links https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
```

Note: If installation with `requirements.txt` does not work, try installing everything in `requirements.txt` separately.

If you run into problem install d2go. Follow the instruction here: https://github.com/facebookresearch/d2go/blob/main/README.md. Use `python setup.py install`.

Prepare Sylph module, run `python setup.py develop`.


# META_ARCHITECTURE
Name: `MetaOneStageDetector`

Specify function to build backbone: `build_fcos_resnet_fpn_backbone`

The architecture constructs the model, including the base detector, backbone, and few-shot hypernetwork.

# Few-shot Hypernetwork Code Structure
## Code generator
To customize or add another code generator, go to directory [`sylph/modeling/code_generator/`](https://github.com/facebookresearch/sylph-few-shot-detection/tree/main/sylph/modeling/code_generator). An example is to add `roi_encoder`.

Please refer to [`build.py`](https://github.com/facebookresearch/sylph-few-shot-detection/blob/main/sylph/modeling/code_generator/build.py) for initializing the code generator, and forward for the inputs needed to run a code generator and the expected outputs.


## Runner
As different code generators likely differ in code generator parameters, it is better to add a new runner for each code generator we add. For this, refer to [`sylph/runner/meta_fcos_roi_encoder_runner.py`](https://github.com/facebookresearch/sylph-few-shot-detection/blob/main/sylph/runner/meta_fcos_roi_encoder_runner.py). This runner should inherit from `MetaFCOSRunner` for training and testing data loaders and logic.


## Datasets
All datasets are located at `/datasets`

Download COCO images and annotations here: https://cocodataset.org/#download. 
```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

### Expected dataset file structure for COCO:
```
coco/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

### Expected dataset file structure for LVIS:
Download annotation from here: https://www.lvisdataset.org/dataset. 
LVIS uses the same image directory as COCO. 
```
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
```
```
coco/
  {train,val}2017/
lvis/
  lvis_v1_{train,val}.json
```
## Config files
All config files are located at `/configs`, to specify a config file, use prefix `sylph://`.

`configs="sylph://LVIS-Meta-FCOS-Detection/Meta_FCOS_MS_R_50_1x.yaml"`

# Sylph Few-shot Detection Training & Evaluation

Under the root directory [`sylph/`](https://github.com/facebookresearch/sylph-few-shot-detection/blob/main/sylph/tools), run `tools/train_net.py`, and specify `--config-file` and `--runner`.


Provide the test mode, the number of steps, the number of GPUs, and the batch size, set to a very small number to test the workflow end to end. Use
`export SYLPH_TEST_MODE=true` to turn it on.

## Train Meta-FCOS

Run at the `root` directory.

### COCO
Pre-training on 60 COCO base classes.
```
python3 tools/train_net.py --runner sylph.runner.MetaFCOSRunner --config-file "sylph://COCO-Detection/Meta-FCOS/Meta-FCOS-pretrain.yaml" --num-processes 3  --output-dir output/meta-fcos/coco/pretrain/
```

Meta-learning on 60 base classes, and meta-test on 20:

First, find the weight checkpoint you want to use from pretraining, and modify `MODEL.WEIGHTS`.
```
python3 tools/train_net.py --runner sylph.runner.MetaFCOSRunner --config-file "sylph://COCO-Detection/Meta-FCOS/Meta-FCOS-finetune.yaml" --num-processes 3  --output-dir output/meta-fcos/coco/meta-train/ MODEL.WEIGHTS output/meta-fcos/coco/pretrain/model_final.pth
```


### LVIS

Pre-training
```
python3 tools/train_net.py --runner sylph.runner.MetaFCOSRunner --config-file "sylph://LVISv1-Detection/Meta-FCOS/Meta-FCOS-pretrain.yaml" --num-processes 3  --output-dir output/meta-fcos/lvis/pretrain/
```

Meta-leraning. Initialize the weights to the `pth` model in the pre-training stage.
```
python3 tools/train_net.py --runner sylph.runner.MetaFCOSRunner --config-file "sylph://LVISv1-Detection/Meta-FCOS/Meta-FCOS-finetune.yaml" --num-processes 3  --output-dir output/meta-fcos/lvis/meta-train/ MODEL.WEIGHTS output/meta-fcos/coco/pretrain/model_final.pth
```
To change the code generator, change the config file and switch to a different Runner (TODO: merge them together to support differnt code generator in the same runner)

## TFA-FCOS Train & Eval

TBD.

# Test
TBD.


