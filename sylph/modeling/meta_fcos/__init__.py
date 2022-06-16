"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

#!/usr/bin/env python3
"""define logics that takes in
query_images
query_features
query_gt_instances
proposals (faster rcnn)
class_codes
support_set_gt_instances

Two steps:
* use class codes to get class logits
* use clas logits and support set gt instances to adapt the loss
"""

from .fcos import MetaFCOS  # noqa
