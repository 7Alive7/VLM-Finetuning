#!/usr/bin/python3.10.14
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import shutil
from transformers import AutoTokenizer, AutoConfig, AutoProcessor, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import random_split
from Pretrain import VLM
from config import VLMConfig
import swanlab
from PIL import Image
import json
from torch.utils.data import Dataset
from typing import List, Dict, Any

# os.environ["SWANLAB_PROJECT"]="omniVLM"

class SFTdataset(Dataset):
    def __init__(self, iamge_path, text_json_path, tokenizer, processor, config):
        super(SFTdataset, self).__init__()
        self.images_path = iamge_path
        self.text_json_path = text_json_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.datas = []

        with open(text_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.datas.append(json.loads(line))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        image_name = sample['image']
        conversations = sample['conversations']
        conv_text = [{"role": "system", "content": 'You are a helpful assistant.'}]
        for turn in conversations:
            if turn['role'] == 'user':
                conv_text.append({"role": "user", "content": turn['content']})
            elif turn['role'] == 'assistant':
                conv_text.append({"role": "assistant", "content": turn['content']})
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
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = "cuda:0"
    config = VLMConfig()
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.llm_path)
    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)

    model = AutoModelForCausalLM.from_pretrained('') # 选择pretain模型的保存路径
    qwen_model = AutoModelForCausalLM.from_pretrained(config.llm_path)
    model.llm.lm_head.load_state_dict(qwen_model.lm_head.state_dict())
    model.to(device)


    print(f'模型参数量为：{sum(p.numel() for p in model.parameters())}')
    print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')


    images_path = '/home/intel/projects/omnivlm/data/sft/sft_images'
    data_path = '/home/intel/projects/omnivlm/data/sft/sft_data.jsonl'

    output_dir = 'save/sft'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    full_dataset = SFTdataset(images_path, data_path, tokenizer, processor, config)
    train_size = int(0.9 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

    args = TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        do_train=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        learning_rate=1e-4,
        num_train_epochs=2,
        eval_strategy='steps',
        eval_steps=200,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=32,
        logging_steps=100,
        report_to='swanlab',
        run_name='vlm_sft',
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )


    swanlab.login(api_key='') # 替换成自己的api_key

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MyDataCollator(tokenizer)
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir=output_dir)
    trainer.save_state()

