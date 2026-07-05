import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MM_SRC = os.path.join(SCRIPT_DIR, "../src")
sys.path.insert(0, MM_SRC)

for p in [
    "/users/haaissa/nanovlm",
    "/iopsstor/scratch/cscs/haaissa/nanovlm",
    "/workspace/nanovlm",
    "/workspace/nanoVLM",
    os.path.join(SCRIPT_DIR, "../../nanovlm"),
    os.path.join(SCRIPT_DIR, "../../nanoVLM"),
]:
    sys.path.insert(0, os.path.abspath(p))

MM_CHECKPOINT = "/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/Dynamic_resize/checkpoint-11500"
NV_CHECKPOINT = (
    "/iopsstor/scratch/cscs/haaissa/nanovlm/checkpoints/"
    "nanoVLM_siglip2-base-patch16-512_512_mp4_SmolLM2-360M-Instruct_"
    "1xGPU_bs16_40000_lr_vision_5e-05-language_5e-05-0.00512_0602-231814/step_10000"
)

# Path to any Arrow dataset folder that has 'texts' and 'images' columns
DATASET_PATH = "/iopsstor/scratch/cscs/haaissa/MultiMeditron_Clean_Arrow"

# Load a real sample from the dataset
def load_real_sample(dataset_path: str, idx: int = 0):
    """
    Load one sample from the Arrow dataset.
    Returns (pil_image, question_str, answer_str).
    """
    from datasets import load_from_disk
    print(f"\n--- Loading sample #{idx} from dataset ---")
    ds = load_from_disk(dataset_path)
    sample = ds[idx]

    # Image
    raw_img = sample["images"][0]  # could be PIL, bytes, or a dict
    if isinstance(raw_img, Image.Image):
        pil_img = raw_img.convert("RGB")
    elif isinstance(raw_img, dict) and "bytes" in raw_img:
        pil_img = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
    elif isinstance(raw_img, bytes):
        pil_img = Image.open(io.BytesIO(raw_img)).convert("RGB")
    else:
        raise ValueError(f"Unknown image type: {type(raw_img)}")

    # Text
    turn = sample["texts"][0]  # first turn
    question = turn.get("user", "Describe this image.")
    answer   = turn.get("assistant", "")

    print(f"  Image size : {pil_img.size}")
    print(f"  Question   : {question[:100]}")
    print(f"  Answer     : {answer[:100]}")
    return pil_img, question, answer


# hf_m4_transform (embedded to avoid import issues on cluster)
def hf_m4_transform(batch):
    new_examples = {"conversations": [], "modalities": []}
    for texts, images in zip(batch.get("texts", []), batch.get("images", [])):
        convs = []
        for turn in texts:
            convs.append({"role": "user",      "content": turn["user"]})
            convs.append({"role": "assistant", "content": turn["assistant"]})

        mods = []
        if images is not None:
            for img in images:
                if img.mode not in ("RGB", "RGBA", "L", "1"):
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                mods.append({"type": "image", "value": {"bytes": buf.getvalue()}})

        if mods and convs:
            content = convs[0]["content"]
            content = content.replace("<|image|>", "").replace("<image>", "").strip()
            convs[0]["content"] = "<|image|>" * len(mods) + content

        new_examples["conversations"].append(convs)
        new_examples["modalities"].append(mods)
    return new_examples


# Attention extraction helpers
def compute_attention_ratio(attn_matrix, img_token_positions, answer_token_positions):
    """
    Compute the fraction of attention that ANSWER tokens direct toward IMAGE tokens.

    attn_matrix          : (seq, seq) average-over-heads attention from the last layer
    img_token_positions  : 1-D tensor / list of indices where image tokens are placed
    answer_token_positions: 1-D tensor / list of indices for the assistant answer tokens

    Returns (attn_to_image, attn_to_text) as floats in [0, 1].
    """
    if len(img_token_positions) == 0:
        print("WARNING: no image token positions found!")
        return 0.0, 1.0
    if len(answer_token_positions) == 0:
        print("WARNING: no answer token positions found!")
        return 0.0, 1.0

    # Rows = answer tokens,  Cols = ALL tokens
    answer_rows = attn_matrix[answer_token_positions, :]          # (n_answer, seq)
    # Each row already sums to ~1.0 (softmax output)
    # Average over answer tokens then sum the image columns
    mean_answer_attn = answer_rows.mean(dim=0)                    # (seq,)
    attn_to_image    = mean_answer_attn[img_token_positions].sum().item()
    attn_to_text     = 1.0 - attn_to_image

    print(f"  Image token count   : {len(img_token_positions)}")
    print(f"  Answer token count  : {len(answer_token_positions)}")
    print(f"  Attn → image        : {attn_to_image*100:.2f}%")
    print(f"  Attn → text         : {attn_to_text*100:.2f}%")
    return attn_to_image, attn_to_text


# MultiMeditron
def get_mm_attentions(pil_img, question, answer):
    from multimeditron.model.model import MultiModalModelForCausalLM, ChatTemplate
    from transformers import AutoTokenizer
    from multimeditron.model.data_loader import DataCollatorForMultimodal
    from multimeditron.dataset.loader import RawImageLoader

    print("\n--- Loading MultiMeditron ---")
    model = MultiModalModelForCausalLM.from_pretrained(
        MM_CHECKPOINT, dtype=torch.float16, trust_remote_code=True
    )
    model.cuda().eval()

    tokenizer    = AutoTokenizer.from_pretrained(MM_CHECKPOINT, trust_remote_code=True)
    
    # [DEBUG] Using qwen3 which is actually ChatML in MultiMeditron
    print("  [DEBUG] MultiMeditron is loading ChatTemplate: 'qwen3'")
    chat_template = ChatTemplate.from_name("qwen3")
    modality_processors = model.processors()
    modality_loaders    = {"image": RawImageLoader()}

    data_collator = DataCollatorForMultimodal(
        tokenizer=tokenizer,
        modality_processors=modality_processors,
        modality_loaders=modality_loaders,
        attachment_token="<|image|>",
        chat_template=chat_template,
        add_generation_prompt=True,
    )

    batch = {
        "texts":  [[{"user": question, "assistant": answer}]],
        "images": [[pil_img]],
    }
    transformed = hf_m4_transform(batch)
    transformed_batch = [
        {"conversations": transformed["conversations"][0],
         "modalities":    transformed["modalities"][0]}
    ]
    collated = data_collator.torch_call(transformed_batch)

    # GPU inputs
    inputs = {
        "input_ids":      collated["input_ids"].cuda(),
        "attention_mask": collated["attention_mask"].cuda(),
        "position_ids":   collated["position_ids"].cuda(),
        "processed_multimodal_inputs": {
            "batch_idx":  {k: v.cuda() for k, v in collated["processed_multimodal_inputs"]["batch_idx"].items()},
            "token_range":{k: v.cuda() for k, v in collated["processed_multimodal_inputs"]["token_range"].items()},
            "stacked":    {k: [t.cuda().to(torch.float16)]
                           for k, img_list in collated["processed_multimodal_inputs"]["stacked"].items()
                           for t in img_list},
        },
    }

    # Image token positions from token_range (ground truth from the collator)
    tr = collated["processed_multimodal_inputs"]["token_range"]
    img_positions = []
    for modality_name, ranges in tr.items():
        ranges = ranges.cpu()
        print(f"  token_range['{modality_name}'] shape: {ranges.shape}, values: {ranges}")
        # token_range is already a flat list of ALL image token positions (shape [n_tokens])
        # NOT [start, end] pairs — just extend directly
        img_positions.extend(ranges.tolist())
    img_positions = torch.tensor(img_positions, dtype=torch.long)


    print(f"\nMultiMeditron – image spans: {img_positions[:5].tolist()}...{img_positions[-5:].tolist()} ({len(img_positions)} tokens)")

    # Answer token positions
    # The assistant answer is everything after the last occurrence of the
    # assistant-start delimiter (we look for token IDs after the last <|im_start|>)
    input_ids_cpu = collated["input_ids"][0]
    im_start_id   = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_start_positions = (input_ids_cpu == im_start_id).nonzero(as_tuple=True)[0]
    # The very last <|im_start|> is the generation prompt ("assistant\n")
    # The second-to-last <|im_start|> marks the beginning of the assistant answer
    if len(im_start_positions) >= 2:
        answer_start = im_start_positions[-2].item() + 1   # skip the delimiter itself
    else:
        answer_start = im_start_positions[-1].item() + 1
    answer_positions = torch.arange(answer_start, len(input_ids_cpu))
    print(f"MultiMeditron – answer tokens [{answer_start}:{len(input_ids_cpu)}]  ({len(answer_positions)} tokens)")

    # [DEBUG] Print decoded string to verify exactly what is fed
    decoded_mm = tokenizer.decode(input_ids_cpu)
    print(f"\n[DEBUG] MultiMeditron Decoded Sequence:\n{repr(decoded_mm)}\n")

    print("Running MultiMeditron Forward Pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            processed_multimodal_inputs=inputs["processed_multimodal_inputs"],
            output_attentions=True,
            return_dict=True,
        )

    # last layer, average over heads → (seq, seq)
    last_layer_attn = outputs.attentions[-1][0].mean(dim=0).cpu()

    print("\nMultiMeditron attention stats:")
    attn_to_img, attn_to_txt = compute_attention_ratio(
        last_layer_attn, img_positions, answer_positions
    )
    return attn_to_img, attn_to_txt, pil_img


# nanoVLM
def get_nv_attentions(pil_img, question, answer):
    import math
    import types
    import torch.nn.functional as F
    from models.vision_language_model import VisionLanguageModel
    from data.datasets import VQADataset
    from data.processors import get_image_processor, get_tokenizer

    print("\n--- Loading nanoVLM ---")
    model = VisionLanguageModel.from_pretrained(NV_CHECKPOINT)
    model.half().cuda().eval()

    tokenizer       = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens, model.cfg.lm_chat_template)
    image_processor = get_image_processor(model.cfg.max_img_size, model.cfg.vit_img_size, model.cfg.resize_to_max_side_len)

    item = {
        "images": [pil_img],
        "texts":  [{"user": question, "assistant": answer}],
    }
    dataset   = VQADataset([item], tokenizer, image_processor, model.cfg.mp_image_token_length)
    processed = dataset[0]

    input_ids_cpu    = processed["input_ids"]
    attention_mask   = processed["attention_mask"].unsqueeze(0).cuda()
    input_ids        = input_ids_cpu.unsqueeze(0).cuda()
    images           = [img.cuda().to(torch.float16) for img in processed["images"]]

    # Image token positions
    img_token_id  = tokenizer.convert_tokens_to_ids("<|image|>")
    img_positions = (input_ids_cpu == img_token_id).nonzero(as_tuple=True)[0]
    print(f"\nnanoVLM – image token id={img_token_id}, found {len(img_positions)} image tokens")
    if len(img_positions):
        print(f"  positions: {img_positions[:5].tolist()}...{img_positions[-5:].tolist()}")

    # Answer token positions
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_start_positions = (input_ids_cpu == im_start_id).nonzero(as_tuple=True)[0]
    if len(im_start_positions) >= 2:
        answer_start = im_start_positions[-2].item() + 1
    else:
        answer_start = im_start_positions[-1].item() + 1
    answer_positions = torch.arange(answer_start, len(input_ids_cpu))
    print(f"nanoVLM – answer tokens [{answer_start}:{len(input_ids_cpu)}]  ({len(answer_positions)} tokens)")

    # [DEBUG] Print decoded string to verify exactly what is fed
    decoded_nv = tokenizer.decode(input_ids_cpu)
    print(f"\n[DEBUG] nanoVLM Decoded Sequence:\n{repr(decoded_nv)}\n")

    # Monkey-patch last block to capture attention
    captured_attentions = []
    from models.language_model import apply_rotary_pos_embd

    def patched_forward(self_attn, x, cos, sin, attn_mask=None, block_kv_cache=None):
        B, T_curr, C = x.size()
        q = self_attn.q_proj(x).view(B, T_curr, self_attn.n_heads,    self_attn.head_dim).transpose(1, 2)
        k = self_attn.k_proj(x).view(B, T_curr, self_attn.n_kv_heads, self_attn.head_dim).transpose(1, 2)
        v = self_attn.v_proj(x).view(B, T_curr, self_attn.n_kv_heads, self_attn.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_embd(q, k, cos, sin)

        k_exp = k.repeat_interleave(self_attn.n_kv_groups, dim=1)
        v_exp = v.repeat_interleave(self_attn.n_kv_groups, dim=1)
        T_kv  = k_exp.size(2)

        # Manual softmax attention (to capture the matrix)
        scale   = math.sqrt(self_attn.head_dim)
        attn_w  = torch.matmul(q.float(), k_exp.float().transpose(2, 3)) / scale
        if T_curr == T_kv and T_curr > 1:
            causal = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_curr)
            attn_w = attn_w.masked_fill(~causal, float("-inf"))
        attn_w = F.softmax(attn_w, dim=-1)

        captured_attentions.append(attn_w.detach().cpu().half())   # save memory

        y = (attn_w.to(v_exp.dtype) @ v_exp).transpose(1, 2).contiguous().view(B, T_curr, C)
        y = self_attn.out_proj(y)
        return y, None

    model.decoder.blocks[-1].attn.forward = types.MethodType(
        patched_forward, model.decoder.blocks[-1].attn
    )

    print("Running nanoVLM Forward Pass...")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, _ = model(input_ids, images, attention_mask=attention_mask)

    # (n_heads, seq, seq) → mean over heads → (seq, seq)
    last_layer_attn = captured_attentions[0][0].float().mean(dim=0)

    print("\nnanoVLM attention stats:")
    attn_to_img, attn_to_txt = compute_attention_ratio(
        last_layer_attn, img_positions, answer_positions
    )
    return attn_to_img, attn_to_txt


# Plot
def plot_attentions():
    print("\n--- Starting Debug Visualization ---")

    pil_img, question, answer = load_real_sample(DATASET_PATH, idx=0)

    mm_img_attn, mm_txt_attn, real_img = get_mm_attentions(pil_img, question, answer)
    nv_img_attn, nv_txt_attn           = get_nv_attentions(pil_img, question, answer)

    print(f"\n{'='*55}")
    print(f"FINAL RESULTS")
    print(f"  MultiMeditron → image: {mm_img_attn*100:.2f}%  |  text: {mm_txt_attn*100:.2f}%")
    print(f"  nanoVLM       → image: {nv_img_attn*100:.2f}%  |  text: {nv_txt_attn*100:.2f}%")
    print(f"{'='*55}\n")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: the real image used
    axes[0].imshow(real_img)
    axes[0].set_title("Input image (from dataset)")
    axes[0].axis("off")

    # Right: attention bar chart
    labels = ["Image Tokens", "Text Tokens"]
    mm_vals = [mm_img_attn * 100, mm_txt_attn * 100]
    nv_vals = [nv_img_attn * 100, nv_txt_attn * 100]
    x     = np.arange(len(labels))
    width = 0.35

    ax = axes[1]
    bars_mm = ax.bar(x - width/2, mm_vals, width, label="MultiMeditron", color="#e74c3c", alpha=0.85)
    bars_nv = ax.bar(x + width/2, nv_vals, width, label="nanoVLM",       color="#2ecc71", alpha=0.85)
    for bar in bars_mm:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontweight="bold", fontsize=10)
    for bar in bars_nv:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontweight="bold", fontsize=10)

    ax.set_ylabel("Attention (%)")
    ax.set_title('Cross-Modal Attention: how much does the LLM look at the image\n(averaged over all answer tokens, last layer, all heads)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, max(max(mm_vals), max(nv_vals)) * 1.25)

    plt.tight_layout()
    plt.savefig("attention_comparison.png", dpi=150)
    print("Saved → attention_comparison.png")


if __name__ == "__main__":
    plot_attentions()

