#!/usr/bin/python3.10.14
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from config import VLMConfig
from train_with_attention_mask import VLM
from PIL import Image


# 初始化配置和模型
config = VLMConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 注册 VLM 用于自动加载
from transformers import AutoConfig
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

# 加载模型和工具
model_path = ""  # 可改为 SFT 模型路径
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(config.llm_path)
processor = AutoProcessor.from_pretrained(config.vision_model_path)
qwen_model = AutoModelForCausalLM.from_pretrained(config.llm_path)
model.llm.lm_head.load_state_dict(qwen_model.lm_head.state_dict())
model.eval()
# 输入图像路径
image_path = input("请输入图像路径（可直接拖拽图片进来）:\n>>> ").strip().strip("'")
image = Image.open(image_path).convert("RGB")
pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

# 对话历史
history = [{"role": "system", "content": "You are a helpful assistant."}]
first_round = True

print(f'LLM load path: {model_path}')

print("\n开始对话（输入 exit 退出）")
while True:
    user_input = input("你：")
    if user_input.strip().lower() in ["exit", "quit"]:
        break

    # 在第一轮对话中，确保在用户输入结尾追加 "\n<image>"
    if first_round:
        if not user_input.endswith("\n<image>"):
            user_input = user_input + "\n<image>"
            current_pixel = pixel_values
        first_round = False
        cached_pixel_values = pixel_values
    else:
        current_pixel = cached_pixel_values

    # 构建 Prompt + 插入 image pad
    history.append({"role": "user", "content": user_input})
    prompt_text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    ).replace("<image>", "<|image_pad|>" * config.image_pad_num)

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=current_pixel,
            max_new_tokens=256,
            temperature=0.7
        )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"🤖：{answer}\n")

    history.append({"role": "assistant", "content": answer})
