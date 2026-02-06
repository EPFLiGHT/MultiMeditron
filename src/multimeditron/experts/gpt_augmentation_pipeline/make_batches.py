from pathlib import Path
import json
import base64
import os
from tqdm import tqdm
from typing import Optional
from utils import load_data
import config
from typing import Literal

SKIN_PROMPT = """You are a multimodal language model tasked with generating structured clinical interpretations of dermatologic images for training datasets. Use the following concise guidelines to ensure clarity, accuracy, and informativeness:

1. Case context and body site:
   - Identify the anatomic location(s) if discernible.
   - Summarize the overall clinical context if present in the text (age group, sex at birth, Fitzpatrick skin type, race/ethnicity, year, and any provided dermatologist differential with weights).

2. Lesion morphology (primary and secondary):
   - Primary: macule, patch, papule, plaque, nodule, vesicle, bulla, pustule, wheal, tumor, erosion, ulcer, fissure.
   - Secondary/surface changes: scale, crust, lichenification, excoriation, atrophy, scar.
   - For each, describe number (single/few/multiple), approximate size (in mm or cm if estimable), shape (round/oval/annular/arciform/serpiginous/polycyclic), borders (well/poorly defined, regular/irregular), symmetry (symmetric/asymmetric), and arrangement (grouped/linear/dermatomal/follicular/photodistributed/acral/flexural/extensor/intertriginous).

3. Color and vascular features:
   - Describe color(s) (erythematous, hyperpigmented, hypopigmented, violaceous, brown, black, skin-colored, yellow) and any color variegation.
   - Note blanching vs. non-blanching if inferable; mention purpura/petechiae vs. erythema when relevant.

4. Distribution and extent:
   - Localized vs. widespread; unilateral vs. bilateral; focal vs. diffuse.
   - Indicate percentage body surface area (rough estimate) only if reasonable.

5. Surface/texture and special signs:
   - Scale (fine/thick/greasy/micaceous), crust, exudate, ulceration, keratotic changes.
   - Palpation-inferable cues (induration, tenderness) only if strongly implied.
   - Name pertinent signs if visible (e.g., Koebner phenomenon, Auspitz sign, Darier sign) but avoid speculation.

6. Dermoscopy (if image quality permits inference):
   - Mention global pattern and key structures (pigment network, dots/globules, streaks, blue-white veil, vessels—dotted, linear, arborizing).
   - State “dermoscopic assessment not possible” if not applicable.

7. Skin type & skin-of-color considerations:
   - If Fitzpatrick skin type or race/ethnicity is provided, discuss how it might impact presentation (erythema visibility, pigment alteration, post-inflammatory dyspigmentation), without making unsupported assumptions.

8. Image quality and limitations:
   - Comment on focus, lighting, color balance, framing, obstructions (hair, dressings), and whether multiple views would help.
   - Explicitly state when key details are not discernible.

9. Differential diagnosis and alignment with provided weights:
   - Provide a short, ranked differential diagnosis consistent with visual findings.
   - If the input includes a dermatologist differential with weights, reconcile your visual assessment with it: indicate agreement, partial agreement, or divergence, and briefly justify.
   - For malignant or urgent possibilities (e.g., melanoma, SCC, rapidly expanding ulcers, extensive cellulitis), explicitly flag as “red flags” and suggest expedited evaluation.

10. Impression/conclusion and next steps (dataset annotation style, not medical advice):
   - Provide a concise impression synthesizing morphology, distribution, and color.
   - Suggest appropriate next steps for dataset labeling (e.g., “request dermoscopic view,” “capture scale and body map,” “obtain close-up with ruler,” “collect symptom duration”).
   - Avoid prescriptive patient management; keep recommendations framed for data curation and annotation quality.

General instructions:
- If features are not visible or not applicable, briefly explain why.
- Avoid speculation; maintain clinical precision.
- Do not use markdown formatting; the enhanced description must start directly with the text of the description (no titles or bullets).
- Keep tone like a dermatologist writing a concise note for dataset curation, not a clinical chart for patient care.

Now here is the info:
-----
"""

OPH_PROMPT = """You are a multimodal language model tasked with generating structured clinical interpretations of ophthalmic images for training datasets. Use the following concise guidelines to ensure clarity, accuracy, and informativeness:

1. Case context and eye information:
   - Identify laterality (OD, OS, OU) and any patient metadata in the text (age, sex, diagnosis labels, eye ID).
   - Note image type or modality if discernible (e.g., color fundus, ultra-widefield [UWF], fluorescein angiography, OCT, fundus autofluorescence).
   - Summarize relevant clinical context such as main diagnosis or screening target (AMD, DR, glaucoma, etc.).

2. Image region and structures:
   - Specify anatomic landmarks visible: optic disc, macula/fovea, vascular arcades, periphery.
   - Mention if image captures posterior pole only or includes mid-peripheral/peripheral retina.

3. Pathologic or notable findings:
   - Describe visible abnormalities (e.g., drusen, hemorrhage, exudate, neovascular membrane, atrophy, pigment mottling, vessel tortuosity, microaneurysm, cotton-wool spot, retinal tear, detachment, vitreous opacity, choroidal lesion).
   - Indicate their approximate location (macular, peripapillary, superior/inferior, nasal/temporal, peripheral) and distribution (focal, multifocal, diffuse, sectoral).
   - Comment on symmetry if both eyes are present.

4. Color, reflectance, and illumination:
   - Note overall color tone (normal reddish-orange, pale, hypopigmented, hyperpigmented).
   - Mention uneven illumination, shadowing, glare, or artifacts affecting interpretation.

5. Optic disc and macula evaluation:
   - Describe disc color, margins, cup-to-disc ratio, and any signs of edema, pallor, or hemorrhage.
   - Assess macular area for integrity of foveal reflex, drusen, hemorrhage, or exudate.

6. Vessels and background:
   - Comment on arteriolar/venular caliber, crossing changes, sheathing, or tortuosity.
   - Describe background retina (normal, granular, atrophic, scarred).

7. Image quality and limitations:
   - Assess focus, contrast, illumination, and field of view.
   - List quality flags (e.g., field of view, contrast, illumination, artifacts, overall quality).
   - State when image quality limits grading.

8. Differential diagnosis and alignment with provided label:
   - Provide a concise, ranked differential consistent with visual findings.
   - Compare with provided diagnosis label: confirm agreement, partial agreement, or disagreement and briefly justify.

9. Impression / annotation summary (for dataset labeling, not clinical care):
   - Summarize key findings succinctly.
   - Suggest next steps for dataset improvement (e.g., “request OCT for confirmation,” “obtain higher-contrast image,” “capture fellow eye,” “verify diagnosis label consistency”).
   - Avoid patient management or therapeutic advice; stay within data-annotation framing.

General instructions:
- Maintain clinical precision but avoid speculation.
- Explicitly note if certain structures or details are not visible.
- Do not use markdown or bullet formatting in the generated description; start directly with prose.
- Keep tone like an ophthalmologist documenting structured findings for dataset curation, not a patient note.

Now here is the info:
-----
"""

SKIN_CONTENT= "You are an expert assistant for dermatology dataset annotation."
OPH_CONTENT= "You are an expert assistant for ophthalmology dataset annotation."

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

MAX_TOKENS = 1000
# 50 MB per batch part is well below OpenAI limits and avoids failures
MAX_PART_SIZE_BYTES = int(0.05 * 1024 ** 3)

TaskType = Literal["skin", "ophthalmology"]

# ---------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------

def build_request(
    *,
    text: str,
    image_b64: str,
    request_id: int,
    task: TaskType,
) -> dict:
    """Build a single OpenAI Batch API request object."""
    if task == "skin":
        system_content = SKIN_CONTENT
        prompt = SKIN_PROMPT
    elif task == "ophthalmology":
        system_content = OPH_CONTENT
        prompt = OPH_PROMPT
    else:
        # Should be unreachable due to typing + _get_env_task validation.
        raise ValueError(f"Unknown task type: {task}")

    return {
        "custom_id": f"request-{request_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": config.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt + f"INITIAL DESCRIPTION: {text}\n-----\nGPT:\n",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": MAX_TOKENS,
        },
    }


# ---------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------


def process_dataset(
    *,
    output_dir: str,
    task: TaskType,
    nb_samples: Optional[int] = None,
) -> None:
    """
    Build batch JSONL files for OpenAI Batch API.

    Writes one or more files: part_1.jsonl, part_2.jsonl, ...
    Splits by MAX_PART_SIZE_BYTES.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    examples, _ = load_data(nb_samples=nb_samples)
    if not examples:
        raise RuntimeError("No examples loaded; aborting batch creation.")

    mode = "ALL" if nb_samples is None else f"{nb_samples}"
    print(f"[INFO] TASK_TYPE={task} | NB_SAMPLES={mode} | Output={output_path}")
    print(f"[INFO] Building batches for {len(examples)} samples")

    part_idx = 1
    bytes_in_part = 0

    part_path = output_path / f"part_{part_idx}.jsonl"
    part_file = part_path.open("w", encoding="utf-8")

    for i, (text, encoded_images) in tqdm(
        enumerate(examples, start=1),
        total=len(examples),
        desc="Building batch requests",
    ):
        if not encoded_images:
            continue

        # Enforce one image per request
        image_b64 = encoded_images[0]

        req = build_request(
            text=text,
            image_b64=image_b64,
            request_id=i,
            task=task,
        )

        line = json.dumps(req, ensure_ascii=False) + "\n"
        size = len(line.encode("utf-8"))

        if bytes_in_part + size > MAX_PART_SIZE_BYTES:
            part_file.close()
            part_idx += 1
            bytes_in_part = 0
            part_path = output_path / f"part_{part_idx}.jsonl"
            part_file = part_path.open("w", encoding="utf-8")

        part_file.write(line)
        bytes_in_part += size

    part_file.close()
    print(f"[DONE] Created {part_idx} batch file(s) in {output_path}")


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------


def main() -> None:
    task = _get_env_task()
    nb_samples = _get_env_nb_samples()
    batches_dir = _get_env_batches_dir()

    process_dataset(
        output_dir=batches_dir,
        task=task,
        nb_samples=nb_samples,
    )


if __name__ == "__main__":
    main()
