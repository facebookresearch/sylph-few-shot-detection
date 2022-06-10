# Few shot detection
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
## Expected dataset structure for COCO:
```
coco/
  annotations/
    instances_{train,val}2014.json
  {train,val}2014/
    # image files that are mentioned in the corresponding json
```

## Expected dataset structure for LVIS:
```
coco/
  {train,val}2017/
lvis/
  lvis_v0.5_{train,val}.json
  lvis_v0.5_train_{freq,common,rare}.json
```

LVIS uses the same images and annotation format as COCO. You can use [split_lvis_annotation.py](split_lvis_annotation.py) to split `lvis_v0.5_train.json` into `lvis_v0.5_train_{freq,common,rare}.json`.
# Train
## Command
Under folder  `sylph/tools`. Run `run.py`. Main change includes: `--config-file`, `--runner`.
### Train Meta-FCOS
#### COCO
Pretraining
```
./run.py     --workflow meta_fcos_e2e_workflow    --config-file "sylph://COCO-Detection/Meta-FCOS/Meta-FCOS-pretrain.yaml"     --entitlement ar_rp_vll     --name "coco_pretraining"     --nodes 1 --num-gpus 8   --gpu-type V100_32G     --output-dir manifold://fai4ar/tree/liyin/few-shot/meta-fcos/test     --run-as-secure-group oncall_fai4ar     --canary --async-val --runner "sylph.runner.MetaFCOSRunner"
```

```
python3 train_net.py --runner sylph.runner.MetaFCOSRunner --config-file "COCO-Detection/Meta-FCOS/Meta-FCOS-pretrain.yaml" --num-processes 3 \\n  --output-dir output/meta-fcos/coco/meta-train/WS_iFSD_imagenet1000x100gt 
```

Meta-learning
```
./run.py     --workflow meta_fcos_e2e_workflow    --config-file "sylph://COCO-Detection/Meta-FCOS/Meta-FCOS-finetune.yaml"     --entitlement ar_rp_vll     --name "coco_meta_learn"     --nodes 4 --num-gpus 4   --gpu-type V100_32G     --output-dir manifold://fai4ar/tree/liyin/few-shot/meta-fcos/test     --run-as-secure-group oncall_fai4ar    --async-val --canary --runner "sylph.runner.MetaFCOSRunner"
```


#### LVIS

Pretraining
```
./run.py     --workflow meta_fcos_e2e_workflow    --config-file "sylph://LVISv1-Detection/Meta-FCOS/Meta-FCOS-pretrain.yaml"     --entitlement ar_rp_vll     --name "lvis_pretraining"     --nodes 1 --num-gpus 8   --gpu-type V100_32G     --output-dir manifold://fai4ar/tree/liyin/few-shot/meta-fcos/test     --run-as-secure-group oncall_fai4ar    --async-val --canary --runner "sylph.runner.MetaFCOSRunner"
```
Meta-leraning. Init the weights to the `pth` model in pretraining stage.
```
./run.py     --workflow meta_fcos_e2e_workflow    --config-file "sylph://LVISv1-Detection/Meta-FCOS/Meta-FCOS-finetune.yaml"     --entitlement ar_rp_vll     --name "lvis_meta_learn"     --nodes 4 --num-gpus 4   --gpu-type V100_32G     --output-dir manifold://fai4ar/tree/liyin/few-shot/meta-fcos/test     --run-as-secure-group oncall_fai4ar     --canary --async-val --runner "sylph.runner.MetaFCOSRunner"
```
To change code generator, change config file and switch to a different Runner (TODO: merge them together to support differnt code generator in the same runner)
### Train Mask RCNN
Switch runner to "sylph.runner.MetaFasterRCNNRunner".

Pretraining
```
./run.py     --workflow meta_fcos_e2e_workflow    --config-file "sylph://LVISv1-Detection/Meta-RCNN/Meta-RCNN-FPN-pretrain.yaml"     --entitlement ar_rp_vll     --name "lvis_pretraining"     --nodes 8 --num-gpus 8   --gpu-type V100_32G     --output-dir manifold://fai4ar/tree/liyin/few-shot/meta-fcos/test     --run-as-secure-group oncall_fai4ar     --canary --async-val --runner "sylph.runner.MetaFasterRCNNRunner"
```
Meta_learning
```
./run.py     --workflow meta_fcos_e2e_workflow    --config-file "sylph://LVISv1-Detection/Meta-RCNN/Meta-RCNN-FPN-finetune.yaml"     --entitlement ar_rp_vll     --name "lvis_meta_learn"     --nodes 4 --num-gpus 4   --gpu-type V100_32G     --output-dir manifold://fai4ar/tree/liyin/few-shot/meta-fcos/test     --run-as-secure-group oncall_fai4ar     --canary --async-val --runner "sylph.runner.MetaFasterRCNNRunner"
```
