# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS


@METRICS.register_module()
class RefSegIoUMetric(BaseMetric):

    def __init__(self, eval_first_text: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.eval_first_text = eval_first_text

    def compute_iou(self, pred_seg, gt_seg):
        i = pred_seg & gt_seg
        u = pred_seg | gt_seg
        return i, u

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_instances']['masks'].bool()
            label = data_sample['gt_masks'].to_tensor(
                pred_label.dtype, pred_label.device).bool()
            if self.eval_first_text:
                pred_label = pred_label[0:1]
            else:
                label = label.repeat(pred_label.shape[0], 1, 1)

            # calculate iou
            i, u = self.compute_iou(pred_label, label)

            bsi = len(pred_label)
            iou = i.reshape(bsi, -1).sum(-1) * 1.0 / u.reshape(bsi, -1).sum(-1)
            iou = torch.nan_to_num_(iou, nan=0.0)
            self.results.append((i.sum(), u.sum(), iou.sum(), bsi))

    def compute_metrics(self, results: list) -> dict:
        results = tuple(zip(*results))
        assert len(results) == 4
        cum_i = sum(results[0])
        cum_u = sum(results[1])
        iou = sum(results[2])
        seg_total = sum(results[3])

        metrics = {}
        metrics['cIoU'] = cum_i * 100 / cum_u
        metrics['mIoU'] = iou * 100 / seg_total
        return metrics