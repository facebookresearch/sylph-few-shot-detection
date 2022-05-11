import math
from collections import defaultdict

import torch
from detectron2.data.samplers import RepeatFactorTrainingSampler


class SupportSetRepeatFactorTrainingSampler(RepeatFactorTrainingSampler):
    """
    Similar to TrainingSampler, but a sample may appear more times than others based
    on its "repeat factor". This is suitable for training on class imbalanced datasets like LVIS.
    """

    @staticmethod
    def repeat_factors_from_category_frequency(
        dataset_dicts, repeat_thresh, num_categories
    ):
        """
        Compute (fractional) per-category repeat factors based on category/annotation frequency.
        The repeat factor for an image is a function of the frequency of the rarest
        category labeled in that image. The "frequency of category c" in [0, 1] is defined
        as the fraction of images in the training set (without repeats) in which category c
        appears.
        See :paper:`lvis` (>= v2) Appendix B.2.

        Args:
            dataset_dicts (list[list[dict]]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
                If the frequency is half of `repeat_thresh`, the image will be
                repeated twice.

        Returns:
            torch.Tensor:
                the i-th element is the repeat factor for the dataset image at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_count = defaultdict(float)
        for i in range(num_categories):
            category_count[i] = len(dataset_dicts[i])
        num_annotations = sum(category_count.values())
        category_freq = defaultdict(float)
        for k, v in category_count.items():
            category_freq[k] = v / num_annotations

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: math.sqrt(repeat_thresh / cat_freq) * category_count[cat_id]
            for cat_id, cat_freq in category_freq.items()
        }
        # smallest category_repeat_factor
        smallest_repeat_factor = min(category_rep.values())

        rep_factors = [v / smallest_repeat_factor for v in category_rep.values()]
        print(f"rep_factors: {rep_factors}")

        return torch.tensor(rep_factors, dtype=torch.float32)
