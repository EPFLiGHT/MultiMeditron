import os
import base64
from io import BytesIO
from tqdm import tqdm

import dspy
from datasets import load_from_disk

# 1. Initialize the LM using Qwen/Qwen3-VL-8B-Instruct
lm = dspy.VLLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=1,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    vllm_kwargs={"limit_mm_per_prompt": {"image": 1}}
)
dspy.settings.configure(lm=lm)

# 2. Define DSPy Signatures to force the specific formatting

class CleanUltrasound(dspy.Signature):
    """Clean and reformat an ultrasound medical draft into a highly structured clinical report using the provided image for reference."""
    
    image: dspy.Image = dspy.InputField(desc="The medical ultrasound image")
    original_text: str = dspy.InputField(desc="Original medical description draft")
    
    visible_organs_and_structures: str = dspy.OutputField(desc="identify visible organs and structures")
    features_of_each_organ_structure: str = dspy.OutputField(desc="features of each organ/structure")
    additional_findings: str = dspy.OutputField(desc="additional findings")
    gray_scale_and_doppler_features: str = dspy.OutputField(desc="gray scale and doppler features")
    dynamic_features: str = dspy.OutputField(desc="dynamic features")
    image_quality_and_limitations: str = dspy.OutputField(desc="image quality and limitations")
    impression_conclusion: str = dspy.OutputField(desc="impression/conclusion")

class CleanXRay(dspy.Signature):
    """Clean and reformat an X-ray medical draft into a highly structured clinical report using the ABCDEFH method, referencing the provided image."""
    
    image: dspy.Image = dspy.InputField(desc="The medical X-Ray image")
    original_text: str = dspy.InputField(desc="Original medical description of an X-ray")
    
    airways: str = dspy.OutputField(desc="A - Airways: look for abnormalities such as hilar adenopathy or enlargement")
    breast_shadows_and_bones: str = dspy.OutputField(desc="B - Breast shadows and Bones: assess for rib fractures, lytic bone lesions, rib crowding suggesting volume loss, or other bony anomalies")
    cardiomediastinal_contour: str = dspy.OutputField(desc="C - Cardiomediastinal contour: examine the cardiac silhouette, chamber size, central structures, including costophrenic angles for pleural effusions")
    diaphragm: str = dspy.OutputField(desc="D - Diaphragm: note position of hemidiaphragms, check for free air under the diaphragm")
    edges: str = dspy.OutputField(desc="E - Edges: inspect for pneumothorax, pleural thickening, plaques, fibrosis, apical findings, extrathoracic tissues")
    fields: str = dspy.OutputField(desc="F - Fields: evaluate lung parenchyma by dividing into upper, mid, lower zones, check symmetry, alveolar air space disease, etc.")
    hilum: str = dspy.OutputField(desc="H - Hilum: document the position, vessels, and junctions, especially superior pulmonary vein and inferior pulmonary artery")


# 3. Define Modules to process and format the text

class UltrasoundFormatter(dspy.Module):
    def __init__(self):
        super().__init__()
        # Using Predict to perform the extraction
        self.extractor = dspy.Predict(CleanUltrasound)
        
    def forward(self, image, text):
        pred = self.extractor(image=image, original_text=text)
        
        # Format strictly with the requested structure
        formatted_report = (
            "visible organs and structures\n"
            f"{pred.visible_organs_and_structures}\n\n"
            "features of each organ/structure\n"
            f"{pred.features_of_each_organ_structure}\n\n"
            "additional findings\n"
            f"{pred.additional_findings}\n\n"
            "gray scale and doppler features\n"
            f"{pred.gray_scale_and_doppler_features}\n\n"
            "dynamic features\n"
            f"{pred.dynamic_features}\n\n"
            "image quality and limitations\n"
            f"{pred.image_quality_and_limitations}\n\n"
            "impression/conclusion\n"
            f"{pred.impression_conclusion}"
        )
        return dspy.Prediction(formatted_text=formatted_report)

class XRayFormatter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(CleanXRay)
        
    def forward(self, image, text):
        pred = self.extractor(image=image, original_text=text)
        
        # Format strictly with ABCDEFH
        formatted_report = (
            "**Clinical Observations (using ABCDEFH)**: Describe each area with precision:\n"
            f"   - **A**irways: {pred.airways}\n"
            f"   - **B**reast shadows and **B**ones: {pred.breast_shadows_and_bones}\n"
            f"   - **C**ardiomediastinal contour: {pred.cardiomediastinal_contour}\n"
            f"   - **D**iaphragm: {pred.diaphragm}\n"
            f"   - **E**dges: {pred.edges}\n"
            f"   - **F**ields: {pred.fields}\n"
            f"   - **H**ilum: {pred.hilum}"
        )
        return dspy.Prediction(formatted_text=formatted_report)

# Instantiate modules (global, created once before processing)
ultrasound_module = UltrasoundFormatter()
xray_module = XRayFormatter()

# 4. Helper: convert PIL Image to dspy.Image via base64
def pil_to_dspy_image(pil_img):
    """Convert a PIL.Image from HuggingFace datasets into a dspy.Image (base64 data URI)."""
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return dspy.Image(url=f"data:image/png;base64,{img_b64}")


# 5. Processing functions (no dataset.map, avoids pickling vLLM)

def process_ultrasound_dataset(dataset):
    """Loop through the dataset and return a list of formatted texts."""
    formatted_texts = []
    for i in tqdm(range(len(dataset)), desc="Cleaning ultrasound"):
        row = dataset[i]
        text_content = row.get("text", "")
        images_list = row.get("modalities_images", [])
        raw_img = images_list[0] if isinstance(images_list, list) and len(images_list) > 0 else None

        if text_content and raw_img is not None:
            try:
                dspy_image = pil_to_dspy_image(raw_img)
                pred = ultrasound_module(image=dspy_image, text=text_content)
                formatted_texts.append(pred.formatted_text)
            except Exception as e:
                print(f"[Row {i}] Error processing ultrasound: {e}")
                formatted_texts.append(None)
        else:
            formatted_texts.append(None)
    return formatted_texts


def process_xray_dataset(dataset):
    """Loop through the dataset and return a list of formatted texts."""
    formatted_texts = []
    for i in tqdm(range(len(dataset)), desc="Cleaning X-ray"):
        row = dataset[i]
        text_content = row.get("text", "")
        images_list = row.get("modalities_images", [])
        raw_img = images_list[0] if isinstance(images_list, list) and len(images_list) > 0 else None

        if text_content and raw_img is not None:
            try:
                dspy_image = pil_to_dspy_image(raw_img)
                pred = xray_module(image=dspy_image, text=text_content)
                formatted_texts.append(pred.formatted_text)
            except Exception as e:
                print(f"[Row {i}] Error processing X-ray: {e}")
                formatted_texts.append(None)
        else:
            formatted_texts.append(None)
    return formatted_texts


def process_datasets():
    # CSCS paths mapped from the user's image
    dataset_paths = {
        "BUSI": "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/BUSI_expert",
        "ct2": "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/ct2_expert",
        "DDTI": "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/DDTI_expert",
        "XR-glob": "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/XR-glob_expert",
        "COVID_US": "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/COVID-US-2026",
        "CT2D-glob": "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/CT2D-glob_expert",
    }

    # Ultrasound datasets (BUSI, ct2, DDTI, COVID_US)
    ultrasound_tasks = ["BUSI", "ct2", "DDTI", "COVID_US"]
    # X-ray datasets
    xray_tasks = ["XR-glob"]

    for name, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"Path does not exist (run on CSCS), skipping {name}: {path}")
            continue

        print(f"\n{'='*60}")
        print(f"Loading {name} dataset from {path}")
        print(f"{'='*60}")
        dataset = load_from_disk(path)
        print(f"  -> {len(dataset)} rows loaded")

        if name in ultrasound_tasks:
            formatted_texts = process_ultrasound_dataset(dataset)
        elif name in xray_tasks:
            formatted_texts = process_xray_dataset(dataset)
        else:
            print(f"No specific format for {name}, skipping.")
            continue

        # Add as a NEW column, preserving original "text"
        dataset = dataset.add_column("formatted_text", formatted_texts)

        output_path = f"{path}_cleaned"
        print(f"Saving cleaned {name} to {output_path}")
        dataset.save_to_disk(output_path)

        success_count = sum(1 for t in formatted_texts if t is not None)
        print(f"  -> {success_count}/{len(formatted_texts)} rows successfully formatted")


if __name__ == "__main__":
    process_datasets()

