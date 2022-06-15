# Sylph: The hypernetwork Framework for No-training Few shot detection
## Install packages
Need python>3.8.

Install virtualenv if its not installed yet.
```
python3 -m pip install --user virtualenv
```
Create
```
python3 -m venv env
```
Activate
```
source env/bin/activate
```

Install dependencies:
```
pip install torch
pip3 install -r requirements.txt --find-links https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
```

Note: if direct one step installation does not work, try to install everything in `requirements.txt` separately.

Install d2go separately. Follow the instruction here: https://github.com/facebookresearch/d2go/blob/main/README.md. Use `python setup.py install`.





# META_ARCHITECTURE
name: `MetaOneStageDetector`
specify function to build backbone: `build_fcos_resnet_fpn_backbone`

The architecture constructs the model, including base detector such as backbone and Few-shot hypernetwork.

# How to add a new few-shot hypernetwork?
## Code generator
To customize or add another code generator, go to directory  `sylph/modeling/code_generator`. One example is to add roi_encoder.

Please refer to build.py for initialize the code generator, and forward for inputs to run a code generator and outputs.

## Runner
As different code generator likely differs in code generator parameters, thus it is better that we add a new runner for each code generator we add. In this, refer to `sylph/runner/meta_fcos_roi_encoder_runner.py`. This runner should inherit from `MetaFCOSRunner` for training and testing data loaders, training and testing logics.

## Load the flow
Specify    `--config-file` and `--runner` in `tools/run.py`.
# Config files
`configs="sylph://LVIS-Meta-FCOS-Detection/Meta_FCOS_MS_R_50_1x.yaml"`

# Prepare data
Download coco images and annotations from here: https://cocodataset.org/#download. 
```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```
## Expected dataset structure for COCO:
```
coco/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

## Expected dataset structure for LVIS:
Download annotation from here: https://www.lvisdataset.org/dataset. 
It uses the same image dir as coco. 
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

# Sylph Few-shot detection Train & Eval
## Command
Under folder  `sylph/tools`. Run `run.py`. Main change includes: `--config-file`, `--runner`.
### Train Meta-FCOS
We provide test mode, where the number of steps, number of gpus, and batch size is set to very small number to test the workflow end to end. Use
`export SYLPH_TEST_MODE=true` to turn it on.
#### COCO
Pretraining on 60 base classes.
```
python3 tools/train_net.py --runner sylph.runner.MetaFCOSRunner --config-file "sylph://COCO-Detection/Meta-FCOS/Meta-FCOS-pretrain.yaml" --num-processes 3  --output-dir output/meta-fcos/coco/pretrain/
```

Meta-learning on 60 base clss, and meta-test on 20

First, find the weight checkpoint you want to use from pretraining, and modify `MODEL.WEIGHTS`.
```
python3 tools/train_net.py --runner sylph.runner.MetaFCOSRunner --config-file "sylph://COCO-Detection/Meta-FCOS/Meta-FCOS-finetune.yaml" --num-processes 3  --output-dir output/meta-fcos/coco/meta-train/ MODEL.WEIGHTS output/meta-fcos/coco/pretrain/model_final.pth
```


#### LVIS

Pretraining
```
python3 tools/train_net.py --runner sylph.runner.MetaFCOSRunner --config-file "sylph://LVISv1-Detection/Meta-FCOS/Meta-FCOS-pretrain.yaml" --num-processes 3  --output-dir output/meta-fcos/lvis/pretrain/
```
Meta-leraning. Init the weights to the `pth` model in pretraining stage.
```
python3 tools/train_net.py --runner sylph.runner.MetaFCOSRunner --config-file "sylph://LVISv1-Detection/Meta-FCOS/Meta-FCOS-finetune.yaml" --num-processes 3  --output-dir output/meta-fcos/lvis/meta-train/ MODEL.WEIGHTS output/meta-fcos/coco/pretrain/model_final.pth
```
To change code generator, change config file and switch to a different Runner (TODO: merge them together to support differnt code generator in the same runner)

# TFA-FCOS Train & Eval
