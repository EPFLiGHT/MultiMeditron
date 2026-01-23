# app.py
import os
import argparse
import logging
import torch
import gradio as gr

from typing import Any, AsyncGenerator, Dict, Generator, List, Union
from transformers import AutoTokenizer
from multimeditron.dataset.loader import FileSystemImageLoader
from multimeditron.model.model import ChatTemplate, MultiModalModelForCausalLM
from multimeditron.model.data_loader import DataCollatorForMultimodal


# ==========================
# Args
# ==========================

parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", required=True)
parser.add_argument("--tokenizer_type", required=False, default="apertus")
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
    chat_template=ChatTemplate.from_name(args.tokenizer_type),
    modality_processors=model.processors(),
    attachment_token=ATTACHMENT_TOKEN,
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
def generate_reply(conversations, modalities, temperature=0.0, max_new_tokens=512, top_p=0.95):
    # single-sample path
    sample = {"conversations": conversations, "modalities": modalities}
    batch = collator([sample])

    # keep everything on model device
    device = next(model.parameters()).device
    batch = _move_to_device(batch, device)

    current_tokens = []

    with torch.autocast("cuda", dtype=torch.bfloat16):
        generator = model.inference_generator(
            batch=batch,
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        for current_token_id in generator:
            current_tokens.append(current_token_id[0].item())
            texts = tokenizer.decode(current_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            yield texts


def map_messages_to_multimeditron_format(messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    mapped_messages = []
    modalities = []
    for message in messages:
        mapped_text = ""
        for content_part in message["content"]:
            match content_part["type"]:
                case "text":
                    mapped_text += content_part["text"]
                case "file":
                    mapped_text += ATTACHMENT_TOKEN
                    modalities.append({
                        "type" : "image",
                        "value" : content_part["file"]["path"]
                    })
                case _:
                    logging.warning(f"Skipping unknown content type {content_part['type']}")

        mapped_messages.append({
            "role" : message["role"],
            "content" : mapped_text
        })

    
    return {
        "conversations": mapped_messages,
        "modalities": modalities
    }

# ==========================
# ChatInterface handler func 
# ==========================
def chat_fn(
    message: Union[str, dict],
    history: List[dict],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Generator[str, None, None] | str:
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
    converted_history = map_messages_to_multimeditron_format(history)
    converted_history["conversations"] +=  [{"role": "user", "content": user_for_model}]

    # modalities per this turn (per-message attachments)
    converted_history["modalities"] += build_modalities(file_paths)

    # generate
    try:
        yield from generate_reply(
            conversations=converted_history["conversations"],
            modalities=converted_history["modalities"],
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
                            container=False)
            new_chat_btn = gr.Button("New Chat", variant="secondary", elem_id="sidebar-newchat")

        # main chat
        with gr.Column(elem_id="main", scale=4, min_width=700):
            with gr.Accordion("Generation Settings ⚙️", open=False, elem_id="gen-settings"):
                temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                max_new_tokens = gr.Slider(16, 2048, value=512, step=16, label="Max New Tokens")

            with gr.Group(elem_id="chat-wrap"):
                ci = gr.ChatInterface(
                    fn=chat_fn,
                    chatbot=gr.Chatbot(
                        height=660,
                        render_markdown=True,
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
