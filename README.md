# 项目简介
本项目借鉴OmniVision多模态大模型架构，不基于任何微调框架从0到1实现了一个多模态大模型的finetune（包括Pre-train和SFT）。

# 模型架构
![image](https://github.com/user-attachments/assets/2d9e0ca3-1049-4cf2-8e6d-51bd44488041)
整体模型架构如上图所示，一共包括3个重要部分：
* **Base LLM**：[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main)
* **Vision encoder**：[siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224/tree/main) 这里用的是siglip的小模型，优点是训练快，显存要求低，缺点是效果没那么好，想要追求更好的效果推荐使用更大规模的siglip模型，需要注意的是使用其他版本的siglip模型需要修改代码中的image_pad_num这个参数（下面会讲如何修改）。
* **Connector**：MLP

这里简单介绍一下这个架构的运行逻辑：Vision encoder先是将输入的image划分为固定的patch（这里patch是16 * 16），然后对每个patch进行embedding从而得到整张image的embedding输入。其次MLP层用于对齐和融合image embedding和text embedding，最终将对齐后的image embedding和text embedding进行拼接（有两种主流的拼接方式，一种是cat，另一种是cross-attention，这里用的是cat）并送入base LLM从而得到输出。

**这里有做一个trick，当然也是借鉴Qwen2-VL系列的做法，就是对得到的image token进行了压缩，也就是架构中的Reshape**。以本项目中为例，每张image的size是224 * 224，patch是16 * 16，所以每张图片对应的token数目是14 * 14 = 196。可以发现图像的token数目远大于text的token数，这会带来高延迟和加大模型训练成本。因此，这里将图像嵌入从 [batch_size, 196, hidden_​​size] 转换为 [batch_size, 49, hidden_​​size*4]。这在不影响模型性能的情况下将标记数量减少了4倍。Qwen2-VL的实验结果表明，这种压缩方法提升了模型性能。这里的49也就是上面所说的image_pad_num，也就是将原来的196压缩了4倍，**所以在更改image_pad_num的时候，需要将其改为原有image token数目压缩相应倍数后得到的最终token数**。

# 模型训练

