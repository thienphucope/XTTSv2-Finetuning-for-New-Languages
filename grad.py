import os
import torch
import gc
import sys
import gradio as gr
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import numpy as np

# --- CẤU HÌNH MẶC ĐỊNH ---
# Cậu sửa lại đường dẫn mặc định ở đây nếu muốn
DEFAULT_CHECKPOINT = r"D:\Ope Watson\Project2\XTTSv2-Finetuning-for-New-Languages\OldModels\dvaethiene32gptcluster1\best_model_20570_pruned.pth"
DEFAULT_CONFIG = r"D:\Ope Watson\Project2\XTTSv2-Finetuning-for-New-Languages\OldModels\dvaethiene32gptcluster1\config.json"
DEFAULT_VOCAB = r"D:\Ope Watson\Project2\XTTSv2-Finetuning-for-New-Languages\OldModels\dvaethiene32gptcluster1\vocab.json"
DEFAULT_SPEAKER_REF = r"ref/chunk_0080.wav" # File mẫu mặc định nếu không upload

# Cấu hình môi trường
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Biến toàn cục chứa model
XTTS_MODEL = None

# --- CÁC HÀM XỬ LÝ ---

def release_memory():
    """Giải phóng VRAM"""
    global XTTS_MODEL
    print("\n🧹 Cleaning memory...")
    if XTTS_MODEL is not None:
        del XTTS_MODEL
        XTTS_MODEL = None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("✨ Memory released!")
    return "Model Unloaded & Memory Cleared!"

def load_model(checkpoint_path, config_path, vocab_path):
    """Load model vào VRAM"""
    global XTTS_MODEL
    
    # Dọn dẹp trước khi load mới
    release_memory()
    
    try:
        print(f"⏳ Loading model from {checkpoint_path}...")
        config = XttsConfig()
        config.load_json(config_path)
        
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            use_deepspeed=False
        )
        model.to(DEVICE)
        XTTS_MODEL = model
        print("✅ Model loaded successfully!")
        return "Model Loaded Successfully!"
    except Exception as e:
        return f"Error loading model: {str(e)}"

def run_tts(
    text, 
    ref_audio, 
    language, 
    temperature, 
    length_penalty, 
    repetition_penalty, 
    top_k, 
    top_p, 
    speed
):
    """Hàm chạy Inference"""
    global XTTS_MODEL
    
    if XTTS_MODEL is None:
        raise gr.Error("Chưa load model! Vui lòng bấm 'Load Model' phía trên.")
    
    if not text:
        raise gr.Error("Chưa nhập text!")

    # Nếu không upload file, dùng file mặc định (cần đảm bảo file tồn tại)
    if ref_audio is None:
        if os.path.exists(DEFAULT_SPEAKER_REF):
            ref_audio = DEFAULT_SPEAKER_REF
            print(f"Using default ref: {ref_audio}")
        else:
            raise gr.Error("Vui lòng upload file Reference Audio hoặc chỉnh lại đường dẫn mặc định.")

    try:
        print(f"🔊 Generating: '{text}'...")
        
        # Tính toán Speaker Latents (mỗi lần chạy đều tính lại để support đổi giọng)
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=ref_audio,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
        )

        # Inference
        out = XTTS_MODEL.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=int(top_k),
            top_p=top_p,
            speed=speed
        )

        # Output format cho Gradio: (sample_rate, numpy_array)
        return (24000, np.array(out["wav"]))

    except Exception as e:
        raise gr.Error(f"TTS Error: {str(e)}")

# --- GIAO DIỆN GRADIO ---

with gr.Blocks(title="XTTS v2 - Fine-tune Tester") as demo:
    gr.Markdown("# 🧪 XTTS v2 Fine-tune Tester")
    gr.Markdown("Tool đơn giản để test model XTTS sau khi fine-tune.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Model Settings")
            chk_input = gr.Textbox(label="Checkpoint Path (.pth)", value=DEFAULT_CHECKPOINT)
            cfg_input = gr.Textbox(label="Config Path (.json)", value=DEFAULT_CONFIG)
            vocab_input = gr.Textbox(label="Vocab Path (.json)", value=DEFAULT_VOCAB)
            
            with gr.Row():
                load_btn = gr.Button("Load Model", variant="primary")
                unload_btn = gr.Button("Unload Model", variant="stop")
            
            status_msg = gr.Label(value="Model not loaded", label="Status")

        with gr.Column(scale=1):
            gr.Markdown("### 2. Inference Settings")
            input_text = gr.Textbox(
                label="Input Text", 
                placeholder="Nhập văn bản cần đọc vào đây...", 
                lines=3,
                value="Xin chào, hôm nay trời đẹp quá nhỉ?"
            )
            
            # Input Reference Audio (Upload hoặc Mic)
            ref_audio_input = gr.Audio(
                label="Reference Audio (Voice Cloning)", 
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            lang_dropdown = gr.Dropdown(
                label="Language", 
                choices=["vi", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"], 
                value="vi"
            )

            with gr.Accordion("Advanced Parameters", open=True):
                temp_slider = gr.Slider(label="Temperature", minimum=0.01, maximum=1.0, step=0.05, value=0.75)
                len_pen_slider = gr.Slider(label="Length Penalty", minimum=0.0, maximum=10.0, step=0.1, value=1.0)
                rep_pen_slider = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=10.0, step=0.1, value=5.0)
                top_k_slider = gr.Slider(label="Top K", minimum=1, maximum=100, step=1, value=50)
                top_p_slider = gr.Slider(label="Top P", minimum=0.01, maximum=1.0, step=0.05, value=0.85)
                speed_slider = gr.Slider(label="Speed", minimum=0.1, maximum=2.0, step=0.1, value=1.0)

            gen_btn = gr.Button("Generate Audio", variant="primary", size="lg")
            output_audio = gr.Audio(label="Generated Result", type="numpy")

    # --- SỰ KIỆN (EVENTS) ---
    load_btn.click(
        fn=load_model,
        inputs=[chk_input, cfg_input, vocab_input],
        outputs=[status_msg]
    )
    
    unload_btn.click(
        fn=release_memory,
        inputs=[],
        outputs=[status_msg]
    )

    gen_btn.click(
        fn=run_tts,
        inputs=[
            input_text, 
            ref_audio_input, 
            lang_dropdown, 
            temp_slider, 
            len_pen_slider, 
            rep_pen_slider, 
            top_k_slider, 
            top_p_slider, 
            speed_slider
        ],
        outputs=[output_audio]
    )

if __name__ == "__main__":
    # Dọn dẹp lúc khởi động
    release_memory()
    demo.launch(share=True) # share=True để tạo link public nếu cần
