"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from d2go.utils.visualization import VisualizationEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog


class FewShotVisualizationEvaluator(VisualizationEvaluator):
    # def __init__(
    #     self,
    #     cfg,
    #     tbx_writer,
    #     dataset_mapper,
    #     dataset_name,
    #     train_iter=None,
    #     tag_postfix=None,
    #     class_codes=None,
    #     metadata=None,
    # ):
    #     super().__init__(
    #         cfg,
    #         tbx_writer,
    #         dataset_mapper,
    #         dataset_name,
    #         train_iter=train_iter,
    #         tag_postfix=tag_postfix,
    #     )
    #     # TODO: fix bug: f270754792
    #     self.class_codes = class_codes
    #     self.metadata = metadata
    #     self.tbx_writer._writer.add_embedding(class_codes, metadata=metadata)

    def _initialize_dataset_dict(self, dataset_name: str) -> None:
        # Enable overriding defaults in case the dataset hasn't been registered.

        self._metadata = MetadataCatalog.get(dataset_name)
        # NOTE: Since there's no GT from test loader, we need to get GT from
        # the dataset_dict, this assumes the test data loader uses the item from
        # dataset_dict in the default way.
        self._dataset_dict = DatasetCatalog.get(dataset_name)
        self._file_name_to_dataset_dict = {
            dic["file_name"]: dic for dic in self._dataset_dict[-1]
        }
