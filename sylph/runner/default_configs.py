def add_base_config(_C):
    # Add hyperparameters to load multidatasets with different categories
    _C.DATASETS.ID_TRAIN = [0]
    _C.DATASETS.ID_TEST = [0]
    # Add for specifying base clsses and novel classes splits while train or eval set is "all" to report AP in different groups
    _C.DATASETS.BASE_CLASSES_SPLIT = ""
    _C.DATASETS.NOVEL_CLASSES_SPLIT = ""
    #
    _C.DATASETS.NUMS_CLASSES = [0]
    _C.MODEL.WEIGHTS_FILTER_BY_MODULE = []  # filter out the  configed modules

    # Evaluation
    _C.TEST.EVAL_PERIOD = 0

    # Freeze control
    _C.MODEL.BACKBONE.FREEZE = False
    _C.MODEL.BACKBONE.FREEZE_EXCLUDE = []
    # Freeze proposal generator by component
    _C.MODEL.PROPOSAL_GENERATOR.OWD = False
    _C.MODEL.PROPOSAL_GENERATOR.FREEZE_CLS_TOWER = False
    _C.MODEL.PROPOSAL_GENERATOR.FREEZE_CLS_LOGITS = False
    # Freeze the box branch
    # both tower and reg is frozen
    _C.MODEL.PROPOSAL_GENERATOR.FREEZE_BBOX_BRANCH = False
    _C.MODEL.PROPOSAL_GENERATOR.FREEZE_BBOX_TOWER = False
    # Freeze the whole proposal generator
    _C.MODEL.PROPOSAL_GENERATOR.FREEZE = False
    # Add for Mask RCNN
    _C.MODEL.ROI_HEADS.FREEZE = False

    # Set seed to have deterministic behavior at data loading
    _C.SEED = -1
    return _C


def add_fcos_config(_C):
    # FCOS
    _C.MODEL.FCOS.BOX_QUALITY = ["ctrness"]
    _C.MODEL.FCOS.IOU_MASK = False
    _C.MODEL.FCOS.CLS_LOGITS_KERNEL_SIZE = 1  # switch from 3 x 3 to 1 x 1
    _C.MODEL.FCOS.L2_NORM_CLS_WEIGHT = False
    return _C


def add_tfa_config(_C):
    _C.MODEL.TFA = type(_C)()
    _C.MODEL.TFA.FINETINE = False  # indicates training stage
    _C.MODEL.TFA.TRAIN_SHOT = 10
    # this is to initalize base classes with pretrained base classes weights
    _C.MODEL.TFA.USE_PRETRAINED_BASE_CLS_LOGITS = True
    # this is for TFA-simplified,
    _C.MODEL.TFA.EVAL_WITH_PRETRAINED_BASE_CLS_LOGITS = False

    return _C


def add_default_meta_learn_config(_C):
    # Config the meta-learner and training stage
    _C.MODEL.META_LEARN = type(_C)()
    _C.MODEL.META_LEARN.EPISODIC_LEARNING = False  # episodic learning?
    _C.MODEL.META_LEARN.SHOT = 5
    # shot on novel categories, -1 is for using all shots
    _C.MODEL.META_LEARN.EVAL_SHOT = 10
    _C.MODEL.META_LEARN.BASE_EVAL_SHOT = 10
    _C.MODEL.META_LEARN.CLASS = 5
    _C.MODEL.META_LEARN.USE_ALL_GTS_IN_BASE_CLASSES = True
    # use this only if the detector is fixed but the last classification layer
    _C.MODEL.META_LEARN.EVAL_WITH_PRETRAINED_CODE = False
    # Num of task is the same as the batch size
    _C.MODEL.META_LEARN.QUERY_SHOT = 1  # per class query image while testing

    # default code generator setup
    _C.MODEL.META_LEARN.CODE_GENERATOR = type(_C)()
    # Freeze code generator
    _C.MODEL.META_LEARN.CODE_GENERATOR.FREEZE = False
    _C.MODEL.META_LEARN.CODE_GENERATOR.DISTILLATION_LOSS_WEIGHT = 0.0
    _C.MODEL.META_LEARN.CODE_GENERATOR.NAME = "CodeGenerator"

    # Control ROI pooler (shared)
    _C.MODEL.META_LEARN.CODE_GENERATOR.ROI_BOX = type(_C)()
    _C.MODEL.META_LEARN.CODE_GENERATOR.ROI_BOX.POOLER_RESOLUTION = 7
    _C.MODEL.META_LEARN.CODE_GENERATOR.ROI_BOX.POOLER_TYPE = "ROIAlignV2"
    # Add a config that allows multiple level of features from FPN
    _C.MODEL.META_LEARN.CODE_GENERATOR.ROI_BOX.FPN_MULTILEVEL_FEATURE = False

    # The number of repeat meta test. Only support coco currently.
    _C.TEST.REPEAT_TEST = 1
    return _C


def add_code_genertor_config(_C):
    # Code-geneator
    _C.MODEL.META_LEARN.CODE_GENERATOR.USE_MASK = True
    _C.MODEL.META_LEARN.CODE_GENERATOR.ALL_MASK = False
    _C.MODEL.META_LEARN.CODE_GENERATOR.MASK_NORM = "GN"
    # _C.MODEL.META_LEARN.CODE_GENERATOR.USE_BKG = True
    _C.MODEL.META_LEARN.CODE_GENERATOR.CONV_L2_NORM = False
    # Bias is always predicted (0 + pred) + priori, but if this is turned off, then it does not use the prediction
    _C.MODEL.META_LEARN.CODE_GENERATOR.USE_BIAS = True
    _C.MODEL.META_LEARN.CODE_GENERATOR.BIAS_L2_NORM = False
    _C.MODEL.META_LEARN.CODE_GENERATOR.TOWER_LAYERS = [["GN", ""]]
    _C.MODEL.META_LEARN.CODE_GENERATOR.CLS_LAYER = [
        "GN", "", 1]  # output kernel size
    _C.MODEL.META_LEARN.CODE_GENERATOR.USE_WEIGHT_SCALE = True  # output kernel size

    _C.MODEL.META_LEARN.CODE_GENERATOR.BIAS_LAYER = []  # in default no bias
    _C.MODEL.META_LEARN.CODE_GENERATOR.WEIGHT_LAYER = []  # in default no bias
    _C.MODEL.META_LEARN.CODE_GENERATOR.SCALE_LAYER = []  # in default no bias
    _C.MODEL.META_LEARN.CODE_GENERATOR.BOX_ON = False
    _C.MODEL.META_LEARN.CODE_GENERATOR.BOX_TOWER_LAYERS = []
    _C.MODEL.META_LEARN.CODE_GENERATOR.BOX_CLS_LAYER = ["", "", 2]
    _C.MODEL.META_LEARN.CODE_GENERATOR.BOX_BIAS_LAYER = []
    _C.MODEL.META_LEARN.CODE_GENERATOR.CONTRASTIVE_LOSS = ""
    _C.MODEL.META_LEARN.CODE_GENERATOR.INIT_NORM_LAYER = False
    _C.MODEL.META_LEARN.CODE_GENERATOR.CLS_REWEIGHT = False
    _C.MODEL.META_LEARN.CODE_GENERATOR.META_WEIGHT = False
    _C.MODEL.META_LEARN.CODE_GENERATOR.META_BIAS = False

    # Per-class scale
    _C.MODEL.META_LEARN.CODE_GENERATOR.USE_PER_CLS_SCALE = False

    # decide if we use max operator to compress class code from the input
    # if we set it to True, also set POST_NORM = "GN" to avoid gradient exploding
    _C.MODEL.META_LEARN.CODE_GENERATOR.COMPRESS_CODE_W_MAX = False
    _C.MODEL.META_LEARN.CODE_GENERATOR.POST_NORM = "GN"

    # The previous feature map channels
    _C.MODEL.META_LEARN.CODE_GENERATOR.IN_CHANNEL = 256
    _C.MODEL.META_LEARN.CODE_GENERATOR.OUT_CHANNEL = 256  # class code channel
    _C.MODEL.META_LEARN.CODE_GENERATOR.USE_DEFORMABLE = False

    return _C


def add_roi_encoder_config(_C):
    _C.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER = type(_C)()
    _C.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.NUM_CONV = 0
    _C.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.CONV_DIM = 256
    _C.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.NORM = ""
    _C.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.NUM_FC = 1
    _C.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.FC_DIM = 256

    _C.MODEL.META_LEARN.CODE_GENERATOR.TRANSFORMER_ENCODER = type(_C)()
    _C.MODEL.META_LEARN.CODE_GENERATOR.TRANSFORMER_ENCODER.LAYERS = 1
    _C.MODEL.META_LEARN.CODE_GENERATOR.TRANSFORMER_ENCODER.HEADS = 8
    _C.MODEL.META_LEARN.CODE_GENERATOR.TRANSFORMER_ENCODER.DROPOUT = 0.1

    _C.MODEL.META_LEARN.CODE_GENERATOR.HEAD = type(_C)()
    _C.MODEL.META_LEARN.CODE_GENERATOR.HEAD.NUM_FC = 1
    _C.MODEL.META_LEARN.CODE_GENERATOR.HEAD.FC_DIM = 512
    _C.MODEL.META_LEARN.CODE_GENERATOR.HEAD.OUTPUT_DIM = 256
    return _C


def add_customized_mask_rcnn_config(_C):
    _C.MODEL.ROI_HEADS.JITTER_MATCH_QUALITY = False

    # Add hyperparameters for focal loss in box head
    # Default focal loss is BCE/2
    _C.MODEL.ROI_BOX_HEAD.FOCAL_LOSS_ALPHA = 0.5
    _C.MODEL.ROI_BOX_HEAD.FOCAL_LOSS_GAMMA = 0.0
    # Scaling factor for loss_cls
    # Default to scale loss_cls to be BCE
    _C.MODEL.ROI_BOX_HEAD.CLS_LOSS_WEIGHT = 2.0

    # Add hyperparameters to load multidatasets with different categories
    # For separate-training on a single dataset, it is recommended to ignore the
    # following three configs but avoid `MultiDatasetMapper` and `MultiDatasetRCNN`.
    # For joint-training on multiple datasets, here are the tips:
    #   1. The length of DATASETS.TRAIN and DATASETS.ID_TRAIN must be the same.
    #   2. The elements in DATASETS.ID_TRAIN should be in [0, D), where D is
    #      the length of DATASETS.NUMS_CLASSES.
    #   3. If the i^th element in DATASETS.ID_TRAIN is j, then the categories in
    #      i^th dataset in DATASETS.TRAIN is mapped to the new category space at
    #      [n(1)+...+n(j-1), n(1)+...+n(j)), n(.) is DATASETS.NUMS_CLASSES element.
    #   4. Things are same for test datasets.
    # It is recommended that DATASETS.NUMS_CLASSES is always same for a model either
    # it is trained or evaluated. Different datasets can sometimes correspond to
    # the same ID as long as they share the same categories. It is also allowed to
    # train (resume) or test on part of the datasets with the specified ID's.
    # _C.DATASETS.ID_TRAIN = [0]
    # _C.DATASETS.ID_TEST = [0]
    # _C.DATASETS.NUMS_CLASSES = [0]

    # Used to evaluate in training process to monitor at real-time
    _C.MODEL.ROI_BOX_HEAD.SCORE_THRESH_TRAIN = 0.05

    # Set the bbox branch to be class agnostic in default
    _C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    return _C
