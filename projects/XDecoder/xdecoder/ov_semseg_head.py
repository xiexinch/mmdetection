import torch
from torch import nn
from mmdet.registry import MODELS
import copy
from torch.nn import functional as F


@MODELS.register_module()
class XDecoderOVSemSegHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 pixel_decoder: nn.Module,
                 transformer_decoder: nn.Module,
                 ignore_value: int = 255,
                 task: str = 'semseg',
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__()
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.num_classes = num_classes
        self.task = task
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        transformer_decoder_ = copy.deepcopy(transformer_decoder)
        transformer_decoder_.update(
            task=task)
        self.predictor = MODELS.build(transformer_decoder_)

    def predict(self, features, batch_data_samples, text_prompts=None, extra={}, rescale=True):
        inter_extra = {}
        if self.task in ['semseg', 'instance', 'panoptic']:
            self.predictor.lang_encoder.get_text_embeddings(text_prompts + ["background"], is_eval=True)
        elif self.task == 'ref-semseg':
            token_info = self.predictor.lang_encoder.get_text_token_embeddings(text_prompts, name='grounding',
                                                                               token=False, norm=False)
            token_emb = token_info['token_emb']
            tokens = token_info['tokens']
            query_emb = token_emb[tokens['attention_mask'].bool()]
            inter_extra['grounding_tokens'] = query_emb[:, None]
            inter_extra['class_emb'] = token_info['class_emb']
        elif self.task == 'retrieval':
            token_info = self.predictor.lang_encoder.get_text_token_embeddings(text_prompts, name='grounding',
                                                                               token=False, norm=True)
            inter_extra['class_emb'] = token_info['class_emb']
        inter_extra.update(extra)

        mask_features, multi_scale_features = self.pixel_decoder(features)

        if 'pred_sem_seg' in batch_data_samples[0]:
            batch_input_shape = batch_data_samples[0].metainfo['batch_input_shape']
            pred_sem_segs = [data_samples.pred_sem_seg.data for data_samples in batch_data_samples]
            pred_sem_segs = torch.stack(pred_sem_segs, dim=0)
            grounding_mask = (pred_sem_segs > 0).float()
            grounding_mask = (1 - F.interpolate(grounding_mask, batch_input_shape, mode='nearest')).bool()
            inter_extra['grounding_mask'] = grounding_mask
        predictions = self.predictor(multi_scale_features, mask_features, extra=inter_extra)
        return predictions
