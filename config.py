#!/usr/bin/python3.10.14
# -*- coding: utf-8 -*-

from transformers import PretrainedConfig

class VLMConfig(PretrainedConfig):
    model_type = 'vlm_model'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_path = r'E:\PycharmProjects\LLM_models\Qwen2.5-0.5B-Instruct'
        self.vision_model_path = r'E:\PycharmProjects\LLM_models\siglip2-base-16-224'
        self.image_pad_num = 49
        self.freeze_vision_model = True  # 是否冻结视觉模型进行训练
        self.freeze_llm = True # 是否冻结语言模型
        self.num_layers_to_unfreeze = 2 # 开放llm的前几层参与训练


