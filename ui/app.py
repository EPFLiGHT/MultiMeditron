# app.py
import os
import argparse
import logging
import torch
import gradio as gr

from typing import List, Union
from transformers import AutoTokenizer
from multimeditron.dataset.loader import FileSystemImageLoader
from multimeditron.model.model import MultiModalModelForCausalLM
from multimeditron.model.data_loader import DataCollatorForMultimodal


# ==========================
# Args
# ==========================
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

# paths for assets/css
APP_DIR = os.path.dirname(__file__)
CSS_PATH = os.path.join(APP_DIR, "assets", "chat.css")
LOGO_PATH = os.path.join(APP_DIR, "assets", "Meditron8B_Logo_with_cube.png")
CUSTOM_CSS = ""
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r") as f:
        CUSTOM_CSS = f.read()

# centered description (markdown)
DESC_MD = """
<div class="desc-center">
This is a demo of <b>Multimeditron-8B</b> multimodal LLM, based on LLaMA-3.1-8B-Instruct and CLIP-ViT-L-14
</div>
"""

# ==========================
# Load tokenizer + model
# ==========================
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
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

try:
    model = MultiModalModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        use_safetensors=True
    ).to("cuda")
except Exception as e:
    raise RuntimeError(
        f"Failed to load model from local path '{model_name}'. "
        f"Check that model-*-of-*.safetensors and model.safetensors.index.json are present. "
        f"Original error: {e}"
    )

if getattr(model, "resize_token_embeddings", None):
    model.resize_token_embeddings(len(tokenizer))
model.eval()

# image loader + collator
loader = FileSystemImageLoader(base_path=os.getcwd())
collator = DataCollatorForMultimodal(
    tokenizer=tokenizer,
    tokenizer_type="llama",
    modality_processors=model.processors(),
    attachment_token_idx=attachment_token_idx,
    add_generation_prompt=True,
    modality_loaders={"image": loader},
)

# ==========================
# Helpers 
# ==========================
def build_modalities(all_image_paths: List[str]):
    """return list[dict] acceptable by the collator, using cluster-local paths."""
    return [dict(type="image", value=p) for p in all_image_paths]

def repeat_attachment_tokens(n: int) -> str:
    """insert one token per image; some stacks expect this consistency."""
    if n <= 0:
        return ""
    return " ".join([ATTACHMENT_TOKEN] * n) + " "

def _move_to_device(batch, device):
    """move nested dict/list tensors to model device."""
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(v, device) for v in batch)
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch

@torch.inference_mode()
def generate_reply(conversations, modalities, temperature=0.7, max_new_tokens=512, top_p=0.95):
    # single-sample path
    sample = {"conversations": conversations, "modalities": modalities}
    batch = collator([sample])

    # keep everything on model device
    device = next(model.parameters()).device
    batch = _move_to_device(batch, device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            batch=batch,
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # robust decode for bs=1
    if isinstance(outputs, torch.Tensor):
        return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return texts[0] if texts else ""


# ==========================
# ChatInterface handler func 
# ==========================
def chat_fn(
    message: Union[str, dict],
    history: List[dict],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    """
    message (dict when multimodal): {"text": str, "files": [paths]}
    history: list[{"role": "user"|"assistant", "content": str}]
    returns assistant text (chatinterface will append & clear input)
    """
    # normalize inputs
    if isinstance(message, dict):
        user_text = (message.get("text") or "").strip()
        files = message.get("files") or []
        file_paths = [getattr(f, "name", f) for f in files]
    else:
        user_text = str(message).strip()
        file_paths = []

    history = history or []

    # build what the model should see (one token per image)
    prefix = repeat_attachment_tokens(len(file_paths))
    user_for_model = f"{prefix}{user_text}" if file_paths else user_text

    # conversations: history + last user turn (attachments handled via modalities)
    convs_for_model = list(history) + [{"role": "user", "content": user_for_model}]

    # modalities per this turn (per-message attachments)
    modalities = build_modalities(file_paths)

    # generate
    try:
        reply = generate_reply(
            conversations=convs_for_model,
            modalities=modalities,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )
    except Exception as e:
        reply = f"[Generation error] {e}"

    return reply


# ==========================
# Layout
# ==========================
with gr.Blocks(css=CUSTOM_CSS, title="Multimeditron Base Chat 🩺") as demo:
    gr.Markdown("# Multimeditron Base Chat 🩺")
    gr.Markdown(DESC_MD)

    with gr.Row(elem_id="app-row"):
        # left sidebar
        with gr.Column(elem_id="sidebar", scale=1, min_width=200):
            logo = gr.Image(value=LOGO_PATH, show_label=False, interactive=False, elem_id="sidebar-logo",
                            container=False, show_download_button=False, show_fullscreen_button=False)
            new_chat_btn = gr.Button("New Chat", variant="secondary", elem_id="sidebar-newchat")

        # main chat
        with gr.Column(elem_id="main", scale=4, min_width=700):
            with gr.Accordion("Generation Settings ⚙️", open=False, elem_id="gen-settings"):
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                max_new_tokens = gr.Slider(16, 2048, value=512, step=16, label="Max New Tokens")

            with gr.Group(elem_id="chat-wrap"):
                ci = gr.ChatInterface(
                    fn=chat_fn,
                    type="messages",
                    chatbot=gr.Chatbot(
                        type="messages",
                        height=660,
                        render_markdown=True,
                        show_copy_button=True,
                    ),
                    textbox=gr.MultimodalTextbox(
                        file_types=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"],
                        file_count="multiple",
                        autofocus=True,
                        placeholder="Ask Multimeditron anything",
                    ),
                    multimodal=True,

                    # pass the sliders defined in the accordion:
                    additional_inputs=[temperature, top_p, max_new_tokens],
                    title=None,
                    description=None,
                )

    # clears chat messages, the internal chatbot history and textbox
    def _clear_chat():
        return [], [], gr.update(value=None)

    new_chat_btn.click(
        _clear_chat,
        outputs=[ci.chatbot, ci.chatbot_state, ci.textbox],
    )




# entry
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
        )
        logging.info("Gradio app exited cleanly.")
    except KeyboardInterrupt:
        logging.info("Received KeyboardInterrupt — shutting down app.")
    except Exception as e:
        logging.exception(f"Gradio app crashed: {e}")
        raise
    finally:
        logging.info("App terminated.")
