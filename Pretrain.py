#!/usr/bin/python3.10.14
# -*- coding: utf-8 -*-

import os
import shutil
from config import VLMConfig
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb, swanlab
from PIL import Image
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any


class VLM(PreTrainedModel):
    # 这里需要注册一下config_class类，不然在用AutoModelForCausalLM加载训练好的VLM时会报错
    config_class = VLMConfig
    def __init__(self, config):
        super(VLM, self).__init__(config)

        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(self.config.llm_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_path)
        self.project_layer = nn.Sequential(
            nn.Linear(self.vision_model.config.vision_config.hidden_size * 4, self.llm.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        )
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        if self.config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):

        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        image_embeddings = self.vision_model.vision_model(pixel_values).last_hidden_state
        b, s, d = image_embeddings.shape
        # 压缩图片token
        image_embeddings = image_embeddings.view(b, -1, d*4)

        # 对齐image和text
        image_features = self.project_layer(image_embeddings)
        text_embeddings = text_embeddings.to(image_features.dtype)

        # 得到最终输入
        input_embeddings = self.merge_image_features_to_text_embeddings(input_ids, text_embeddings, image_features)

        outputs = self.llm(inputs_embeds=input_embeddings, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fc = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fc(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, pixel_values=None, **generate_kwargs):
        # 自定义 generate 函数，确保图像信息嵌入后参与整轮 token 的生成。

        if input_ids is None or pixel_values is None:
            raise ValueError("Both input_ids and pixel_values are required for multimodal generation.")

        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        image_embeddings = self.vision_model.vision_model(pixel_values).last_hidden_state
        b, s, d = image_embeddings.shape
        image_embeddings = image_embeddings.view(b, -1, d * 4)
        image_features = self.project_layer(image_embeddings)
        text_embeddings = text_embeddings.to(image_features.dtype)
        input_embeddings = self.merge_image_features_to_text_embeddings(input_ids, text_embeddings, image_features)

        #构建 attention_mask（如果没给）
        if attention_mask is None:
            attention_mask = torch.ones(input_embeddings.shape[:2], dtype=torch.long, device=input_embeddings.device)

        outputs = self.llm.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            **generate_kwargs  # 支持 max_new_tokens, temperature 等常规参数
        )
        return outputs

    def merge_image_features_to_text_embeddings(self, input_ids, text_embeddings, image_features):

        batch, patch, hidde_dim = image_features.shape
        # 找出input_ids中被<image_pad>占位的索引
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        # 将image_features替换原来的<image_pad>的embedding
        text_embeddings[batch_indices, image_indices] = image_features.view(-1, hidde_dim)

        return text_embeddings


class Mydataset(Dataset):
    def __init__(self, iamge_path, text_json_path, tokenizer, processor, config):
        super(Mydataset, self).__init__()
        self.images_path = iamge_path
        self.text_json_path = text_json_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        with open(text_json_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        image_name = sample['image']
        conversations = sample['conversations']
        conv_text = [{"role": "system", "content": 'You are a helpful assistant.'}]
        for turn in conversations:
            if turn['from'] == 'human':
                conv_text.append({"role": "user", "content": turn['value']})
            elif turn['from'] == 'gpt':
                conv_text.append({"role": "assistant", "content": turn['value']})
        image_pad = '<|image_pad|>' * self.config.image_pad_num
        all_text = self.tokenizer.apply_chat_template(conv_text, tokenize=False, add_generation_prompt=False).replace(
            '<image>', image_pad)
        inputs = self.tokenizer(all_text, return_tensors='pt', padding=False)
        input_ids = inputs['input_ids'][0].tolist()

        # 构造attention_mask
        attention_mask = [1] * len(input_ids)

        # 构造 labels：只对 assistant 的 response 位置计算 loss，其它位置用 pad_token_id
        labels = [self.tokenizer.pad_token_id] * len(input_ids)
        result_index = self._find_assistant_token(self.tokenizer, input_ids)
        for i in result_index:
            labels[i[0]:i[1]] = input_ids[i[0]:i[1]]

        # 偏移
        input_ids = input_ids[:-1]
        labels = labels[1:]
        attention_mask = attention_mask[:-1]

        image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
        pixel_values = self.processor(text=None, images=image)['pixel_values'][0]


        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values
        }

    def _find_assistant_token(self, tokenizer, input_ids):
        result = []
        start_index = 0
        end_index = 0
        while start_index <= len(input_ids) - 1:
            if input_ids[start_index] != tokenizer('assistant')['input_ids'][0]:
                start_index += 1
                end_index += 1
            else:
                end_index += 1
                if input_ids[end_index] == tokenizer('<|im_end|>')['input_ids'][0]:
                    result.append((start_index + 1, end_index + 1))
                    start_index = end_index + 1
        return result

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(f['input_ids']) for f in features)
        input_ids, labels, attention_mask, pixel_values = [], [], [], []
        for f in features:
            pad_len = max_len - len(f['input_ids'])
            input_ids.append(f['input_ids'] + [self.tokenizer.pad_token_id] * pad_len)
            labels.append(f['labels'] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(f['attention_mask'] + [0] * pad_len)
            pixel_values.append(f['pixel_values'])
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'pixel_values': torch.stack(pixel_values)
        }

if __name__ == '__main__':
    config = VLMConfig()
    model = VLM(config).cuda()
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    images_path = './pre_train_dataset/image/images'
    data_path = './pre_train_dataset/text/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    output_dir = 'save/pretrain_0718'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=1,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='swanlab',
        run_name='vlm_pretrain_2025_07_18',
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )


    swanlab.login(api_key='') # 替换为你自己的api_key

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Mydataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
    trainer.save_state()
