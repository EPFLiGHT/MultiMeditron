from typing import Dict, List, Tuple, Union

#imports for BiomedCLIP
from open_clip import create_model_from_pretrained, get_tokenizer
from open_clip.model import CustomTextCLIP

from PIL import Image
from tqdm import tqdm

from transformers import VisionTextDualEncoderModel, AutoImageProcessor, AutoTokenizer
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

import json
import os
import torch

DataRow = Dict[str, Union[str, List[Dict[str, str]]]]

image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32") #image processor for CLIP-based models
biomed_processor = [None] #will be filled when loading BiomedCLIP, has its own processor

def load_model(model_path: str) -> VisionTextDualEncoderModel:
    """
    Load the VisionTextDualEncoderModel from the provided path.

    Args:
      model_path: the path to the model
    
    Returns the corresponding VisionTextDualEncoderModel instance.
    """
    
    if model_path == "biomedclip":
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        biomed_processor[0] = preprocess

        return model
    elif model_path == "openai/clip-vit-base-patch32": #base CLIP
        return VisionTextDualEncoderModel.from_vision_text_pretrained(model_path, "FacebookAI/roberta-base")
    else:
        #find the latest checkpoint
        if "checkpoint" not in model_path:
            epochs = []
            for folder in os.listdir(model_path):
                if "checkpoint" in folder:
                    epochs.append(int(folder.split("-")[1]))
            epoch = max(epochs)

            return VisionTextDualEncoderModel.from_pretrained(os.path.join(model_path, f"checkpoint-{epoch}/"))
        else:
            return VisionTextDualEncoderModel.from_pretrained(model_path)

# Preprocessing pipeline for images
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        "`x` should be an instance of `PIL.Image.Image`"
        with torch.no_grad():
            x = self.transforms(x)
        return x
            
image_transformations = Transform(
    224, image_processor.image_mean, image_processor.image_std
)
image_transformations = torch.jit.script(image_transformations)
img_transform = lambda img_path: image_transformations(read_image(img_path, mode=ImageReadMode.RGB))

def encode_img(model: VisionTextDualEncoderModel, img_path: str) -> torch.Tensor:
    """
    Encode an image given a model.

    Args:
      model, the instance of VisionTextDualEncoderModel to use
      img_path, the absolute (or relative) path to the image that should be encoded

    Returns the tensor of the embedding of the image.
    """
    
    device = next(model.parameters()).device

    with torch.no_grad():
        if isinstance(model, CustomTextCLIP):
            preprocess = biomed_processor[0]
            if preprocess is None:
                raise RuntimeError("Did not load the preprocessor of biomedclip. Please run first `load_model('biomedclip')`")
            images = torch.stack([preprocess(Image.open(img_path))]).to(device)
            image_embed = model.encode_image(images)
        elif isinstance(model, VisionTextDualEncoderModel):
            pixel_values = torch.stack([img_transform(img_path)]).to(device)
            vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
            pooled = vision_outputs.pooler_output
            if pooled is None:
                raise RuntimeError("vision_model returned no pooler_output")
            image_embed = model.visual_projection(pooled)
        else:
            raise TypeError(f"Unsupported model type for encode_img: {type(model)}")

    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return image_embed[0]

def preprocess_dataset(data_lines: List[Dict], dataset_path: str) -> List[torch.Tensor]:
    """
    Preprocess a dataset (find the image and apply the image processor on it).

    Args:
      data_lines, the list of examples of the dataset {"text": text, "modalities": [{"type": type, "value": path_to_image}]}
      dataset_path, the absolute (or relative) path to the dataset

    Returns the list of tensors representing each image of the dataset.
    """
    
    def adapt_line(line: DataRow):
        img_path = os.path.join(dataset_path, line["modalities"][0]["value"])
        return img_transform(img_path)

    return [adapt_line(line) for line in tqdm(data_lines)]

def preprocess_dataset_biomed(data_lines: List[Dict], dataset_path: str) -> List[torch.Tensor]:
    """
    Preprocess a dataset for BiomedCLIP (find the image and apply the image processor on it).

    Args:
      data_lines, the list of examples of the dataset {"text": text, "modalities": [{"type": type, "value": path_to_image}]}
      dataset_path, the absolute (or relative) path to the dataset

    Returns the list of tensors representing each image of the dataset.
    """

    preprocess = biomed_processor[0]
    if preprocess is None: #in case biomedclip (and its image processor) had not already been loaded
        load_model("biomedclip")
        preprocess = biomed_processor[0]
    
    def adapt_line(line: DataRow):
        img_path = os.path.join(dataset_path, line["modalities"][0]["value"])
        return preprocess(Image.open(img_path))

    return [adapt_line(line) for line in tqdm(data_lines)]

def make(model: VisionTextDualEncoderModel, dataset: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make the embeddings for a whole dataset, by a given model.

    Args:
      model, the instance of VisionTextDualEncoderModel to use
      dataset, the list of tensors of pre-processed images (or one tensor stacking all)
    """
    
    if not isinstance(dataset, list):
        dataset = [dataset]

    pixel_values = torch.stack(dataset)
    with torch.no_grad():
        if not isinstance(model, CustomTextCLIP):
            image_embeds = model.get_image_features(pixel_values)
        else:
            image_embeds = model.encode_image(pixel_values)

    # Normalization
    image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds