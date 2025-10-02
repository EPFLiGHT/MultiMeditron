# app.py
import os
import argparse
import logging
import torch
import gradio as gr
from typing import List, Tuple

from transformers import AutoTokenizer
from multimeditron.dataset.preprocessor.modality_preprocessor import ModalityRetriever
from multimeditron.dataset.registry.fs_registry import FileSystemImageRegistry
from multimeditron.model.model import MultiModalModelForCausalLM
from multimeditron.model.data_loader import DataCollatorForMultimodal

# ==========================
# Args
# =========================
default_model = "/capstor/store/cscs/swissai/a127/homes/theoschiff/models/MultiMeditron-8B-Clip/checkpoint-813"

parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", required=False, default=default_model)
parser.add_argument("--base_path", required=False, default=os.getcwd(),
                    help="Base path for FileSystemImageRegistry; where your data/images live on the cluster")
parser.add_argument("--share", action="store_true", help="Gradio share link (use cautiously on cluster)")
parser.add_argument("--server_port", type=int, default=7860)
parser.add_argument("--server_name", type=str, default="0.0.0.0")
args, _ = parser.parse_known_args()

ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"
model_name = args.model_checkpoint

# for local-only loading 
LOCAL_ONLY_KW = dict(local_files_only=True)

# ==========================
# Load tokenizer + model
# ==========================
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", **LOCAL_ONLY_KW)
except Exception as e:
    raise RuntimeError(
        f"Failed to load tokenizer from local path '{model_name}'. "
        f"Make sure tokenizer.json/tokenizer_config.json/special_tokens_map.json exist. "
        f"Original error: {e}"
    )

tokenizer.pad_token = tokenizer.eos_token
special_tokens = {"additional_special_tokens": [ATTACHMENT_TOKEN]}
tokenizer.add_special_tokens(special_tokens)
attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)

use_device_map = torch.cuda.is_available() and torch.cuda.device_count() > 1

try:
    model = MultiModalModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        use_safetensors=True,
        device_map="auto" if use_device_map else None,
        **LOCAL_ONLY_KW,
    )
except Exception as e:
    raise RuntimeError(
        f"Failed to load model from local path '{model_name}'. "
        f"Check that model-*-of-*.safetensors and model.safetensors.index.json are present. "
        f"Original error: {e}"
    )

if not use_device_map:
    model = model.to("cuda")

if getattr(model, "resize_token_embeddings", None):
    model.resize_token_embeddings(len(tokenizer))

# Modality + collator utilities
modality_retriever = ModalityRetriever(
    registry=FileSystemImageRegistry(base_path=args.base_path)
)

collator = DataCollatorForMultimodal(
    tokenizer=tokenizer,
    tokenizer_type="llama",
    modality_processors=model.processors(),
    attachment_token_idx=attachment_token_idx,
    add_generation_prompt=True
)

# ==========================
# Helpers
# ==========================
def build_modalities(all_image_paths: List[str]):
    """Return list[dict] acceptable by the collator, using cluster-local paths."""
    return [dict(type="image", value=p) for p in all_image_paths]

@torch.inference_mode()
def generate_reply(conversations, modalities, temperature=0.7, max_new_tokens=512, top_p=0.95):
    sample = {"conversations": conversations, "modalities": modalities}
    sample = modality_retriever.merge_modality_with_sample(sample)
    batch = collator([sample]) 

    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            batch=batch,
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True,
            max_new_tokens=int(max_new_tokens)
        )
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return text

# ==========================
# Gradio app
# ==========================
with gr.Blocks(title="Multimeditron Chat") as demo:
    gr.Markdown("# Multimeditron Base Chat 🩺 \nUpload images any time and chat with the model.")

    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(
                label="Conversation",
                height=500,
                type="messages"  # (user, assistant)
            )
            user_input = gr.Textbox(placeholder="Type your message…", lines=3, autofocus=True)
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("New Chat")

        with gr.Column(scale=2):
            gr.Markdown("### Images")
            images = gr.File(
                label="Add images (they stay attached across turns until cleared)",
                file_types=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"],
                file_count="multiple"
            )
            current_images_gallery = gr.Gallery(label="Currently attached images", columns=3, height=300)
            remove_images_btn = gr.Button("Clear Attached Images")

            with gr.Accordion("Generation Settings", open=False):
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                max_new_tokens = gr.Slider(16, 2048, value=512, step=16, label="Max New Tokens")

    # states: list of image paths & chat history
    state_images = gr.State([])   # List[str]
    state_history = gr.State([])   # List[{"role": "user"|"assistant", "content": str}]

    def on_image_upload(new_files, img_state):
        # accept new img and append to state
        new_paths = []
        if new_files:
            for f in (new_files if isinstance(new_files, list) else [new_files]):
                # gradio passes temp file paths; we can use them directly if our registry's base_path allows absolute paths.
                new_paths.append(f.name if hasattr(f, "name") else f)
        combined = (img_state or []) + new_paths
        # gallery expects list of (path) or (path, caption)
        gallery_items = [(p, os.path.basename(p)) for p in combined]
        return combined, gallery_items

    def on_clear_images():
        return [], []

    def on_clear_chat():
        return [], [], [], []  # chat, history state, image state, gallery

    def on_send(message, chat_hist, img_paths, temp, topp, mnt):
        # chat_hist is a list of {"role": "...", "content": "..."}
        chat_hist = chat_hist or []
        img_paths = img_paths or []

        if not message or not message.strip():
            return gr.update(), chat_hist, img_paths

        # Prepend ATTACHMENT_TOKEN if images are attached
        user_text = f"{ATTACHMENT_TOKEN} {message}" if img_paths else message

        # 1) add the user message (messages format)
        new_hist = chat_hist + [{"role": "user", "content": user_text}]

        # 2) build modalities & generate
        modalities = build_modalities(img_paths)
        try:
            reply = generate_reply(
                conversations=new_hist,     # already messages-format
                modalities=modalities,
                temperature=temp,
                max_new_tokens=mnt,
                top_p=topp
            )
        except Exception as e:
            reply = f"[Generation error] {e}"

        # 3) append assistant message
        new_hist.append({"role": "assistant", "content": reply})

        # Return to: Chatbot, history state, images state
        return new_hist, new_hist, img_paths



    # Callbacks
    images.upload(on_image_upload, inputs=[images, state_images], outputs=[state_images, current_images_gallery])
    remove_images_btn.click(on_clear_images, outputs=[state_images, current_images_gallery])
    send_btn.click(
        on_send,
        inputs=[user_input, state_history, state_images, temperature, top_p, max_new_tokens],
        outputs=[chat, state_history, state_images]
    )
    user_input.submit(
        on_send,
        inputs=[user_input, state_history, state_images, temperature, top_p, max_new_tokens],
        outputs=[chat, state_history, state_images]
    )
    clear_btn.click(on_clear_chat, outputs=[chat, state_history, state_images, current_images_gallery])


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Launching Multimeditron UI…")
    logging.info(f"Model path: {model_name}")
    logging.info(f"Server: {args.server_name}:{args.server_port}  |  share={args.share}")
    try:
        demo.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
            show_error=True,
            prevent_thread_lock=False,   # keeps the process alive
        )
        logging.info("Gradio app exited cleanly.")
    except KeyboardInterrupt:
        logging.info("Received KeyboardInterrupt — shutting down app.")
    except Exception as e:
        logging.exception(f"Gradio app crashed: {e}")
        raise
    finally:
        logging.info("App terminated.")



