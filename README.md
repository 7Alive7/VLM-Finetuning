# 项目简介
本项目借鉴OmniVision多模态大模型架构，不基于任何微调框架从0到1实现了一个多模态大模型的finetune（包括Pre-train、SFT和DPO）。

# 模型架构
![image](https://github.com/user-attachments/assets/2d9e0ca3-1049-4cf2-8e6d-51bd44488041)
整体模型架构如上图所示，一共包括3个重要部分：
* **Base LLM**：[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main)
* **Vision encoder**：[siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224/tree/main) 这里用的是siglip2的小模型，优点是训练快，显存要求低，缺点是效果没那么好，想要追求更好的效果推荐使用更大规模的siglip模型，需要注意的是使用其他版本的siglip模型需要修改代码中的image_pad_num这个参数。
* **Connector**：MLP

这里简单介绍一下这个架构的运行逻辑：Vision encoder先是将输入的image划分为固定的patch（这里patch是16 * 16），然后对每个patch进行embedding从而得到整张image的embedding输入。其次MLP层用于对齐和融合image embedding和text embedding，最终将对齐后的image embedding和text embedding进行拼接（有两种主流的拼接方式，一种是cat，另一种是cross-attention，这里用的是cat）并送入base LLM从而得到输出。

# 数据来源
## Pre-train
* **Image data**：[liuhaotian/LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/tree/main)
*  **Text data**: [LinkSoul/Chinese-LLaVA-Vision-Instructions](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions/tree/main)



