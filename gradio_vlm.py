import torch
import gradio as gr
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoConfig
from config import VLMConfig
from train_with_attention_mask import VLM
from PIL import Image

# 注册自定义模型
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

# 初始化全局变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = VLMConfig()

# 这里把路径添加上即可
MODEL_PATHS = {
    "SFT模型": "",
    "Pretrain模型": ""
}


current_model = None
tokenizer = None
processor = None

# 加载模型
def load_model(model_choice):
    global current_model, tokenizer, processor
    path = MODEL_PATHS[model_choice]
    current_model = AutoModelForCausalLM.from_pretrained(path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.llm_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)

    # 替换 lm_head
    qwen_model = AutoModelForCausalLM.from_pretrained(config.llm_path)
    current_model.llm.lm_head.load_state_dict(qwen_model.lm_head.state_dict())
    current_model.eval()

# 聊天主逻辑
def chat(user_input, image, history_json, model_choice, temperature, top_p, pixel_values_state):
    if current_model is None:
        load_model(model_choice)

    if history_json is None or len(history_json) == 0:
        history = [{"role": "system", "content": "You are a helpful assistant."}]
        first_round = True
    else:
        history = history_json
        first_round = False

    # 图像处理逻辑
    if first_round:
        if not user_input.endswith("\n<image>"):
            user_input = user_input + "\n<image>"
        image = image.convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        pixel_values_state = pixel_values  # 缓存视觉特征
    else:
        pixel_values = pixel_values_state  # 复用之前的视觉特征

    history.append({"role": "user", "content": user_input})
    prompt_text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    ).replace("<image>", "<|image_pad|>" * config.image_pad_num)

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = current_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=256,
            temperature=temperature,
            top_p=top_p
        )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    history.append({"role": "assistant", "content": answer})

    messages = [{"role": item["role"], "content": item["content"]} for item in history[1:]]
    return "", history, messages, pixel_values_state

# 清除历史函数
def clear_history():
    return [], [], "", None, None  # history, chatbot, input, image, pixel_values_state

# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 多模态对话 Demo (仅首轮上传图像)")

    with gr.Row():
        model_choice = gr.Radio(["SFT模型", "Pretrain模型"], label="选择使用模型", value="SFT模型")

    with gr.Row():
        temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
        top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="上传图像 (仅第一轮使用)")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="多轮对话", type="messages")
            user_input = gr.Textbox(label="你的问题", placeholder="输入文本内容...")
            send_button = gr.Button("发送")
            clear_button = gr.Button("清除历史记录")

    history_state = gr.State([])
    pixel_values_state = gr.State(None)

    send_button.click(
        fn=chat,
        inputs=[user_input, image_input, history_state, model_choice, temperature, top_p, pixel_values_state],
        outputs=[user_input, history_state, chatbot, pixel_values_state]
    )

    clear_button.click(
        fn=clear_history,
        inputs=[],
        outputs=[history_state, chatbot, user_input, image_input, pixel_values_state]
    )

if __name__ == '__main__':
    demo.launch(share=False, server_name="0.0.0.0", server_port=7777)
