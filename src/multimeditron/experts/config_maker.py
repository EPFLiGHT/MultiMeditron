import itertools
import yaml
import os
import sys
from pydantic import BaseModel, Field
from typing import Dict, List, Any

# Define Pydantic models for validation
class Datamix(BaseModel):
    dataset_configs: List[Dict[str, Any]] = Field(
        default=[
            {
                "combined_dataset_MRI": {
                    "dataset_path": "/IRM_jsonl/IRM-train.jsonl",
                    "image_column": "modalities",
                    "caption_column": "text",
                    "weight": 1,
                }
            }
        ]
    )

class BaseConfig(BaseModel):
    learning_rate: float = 5.0e-4
    warmup_steps: int = 2000
    lr_scheduler_type: str = "cosine"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1.0e-6
    weight_decay: float = 0.2
    num_train_epochs: int = 32

class CommonConfig(BaseModel):
    output_dir: str = "./models/"
    vision_model_name: str = "openai/clip-vit-base-patch32"
    text_model_name: str = "naver/splade-v3"
    remove_unused_columns: bool = False
    do_train: bool = True
    per_device_eval_batch_size: int = 64
    dataloader_drop_last: bool = True
    overwrite_output_dir: bool = True
    save_steps: int = 150
    fp16: bool = True
    bf16: bool = True

class Configurations(BaseModel):
    datamixes: Dict[str, Datamix] = Field(
        default={
            "combined_dataset_MRI": Datamix()
        }
    )
    base_configs: Dict[str, BaseConfig] = Field(
        default={
            "initial": BaseConfig(),
            "fine_tuning": BaseConfig(
                learning_rate=1.0e-4,
                warmup_steps=1000,
                lr_scheduler_type="linear",
                weight_decay=0.1,
                num_train_epochs=20,
            ),
            "aggressive_training": BaseConfig(
                learning_rate=1.0e-3,
                warmup_steps=500,
                adam_beta1=0.85,
                adam_beta2=0.95,
                weight_decay=0.3,
                num_train_epochs=40,
            ),
            "regularization_focused": BaseConfig(
                learning_rate=2.5e-4,
                warmup_steps=2000,
                adam_beta1=0.95,
                adam_beta2=0.999,
                weight_decay=0.4,
                num_train_epochs=32,
            ),
        }
    )
    param_ranges: Dict[str, List[Any]] = Field(
        default={
            "learning_rate": [1.0e-4, 5.0e-4, 1.0e-3],
            "num_train_epochs": [40],
        }
    )
    common_config: CommonConfig = Field(default=CommonConfig())

# Load configurations from external YAML file
def load_configurations(config_path: str) -> Configurations:
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    return Configurations(**config_data)

def main(config_path: str):
    # Load configurations
    configs = load_configurations(config_path)

    # Create output directory
    output_dir = "configurations"
    os.makedirs(output_dir, exist_ok=True)

    # Generate configurations
    for datamix_name, datamix in configs.datamixes.items():
        for config_name, base_config in configs.base_configs.items():
            # Create grid search combinations for specified parameters
            param_names = configs.param_ranges.keys()
            param_values = configs.param_ranges.values()
            grid_combinations = list(itertools.product(*param_values))
            
            for idx, combination in enumerate(grid_combinations):
                new_config = base_config.dict()
                new_config.update(dict(zip(param_names, combination)))

                # Add common configuration fields
                common_config = configs.common_config.dict()
                common_config["output_dir"] = f"./models/{datamix_name}_{config_name}_config_{idx + 1}"
                new_config.update(common_config)

                # Save each configuration as a YAML file
                config_filename = f"{datamix_name}_{config_name}_config_{idx + 1}.yaml"
                config_filepath = os.path.join(output_dir, config_filename)

                with open(config_filepath, "w") as f:
                    yaml.dump({"datamix": datamix.dict(), **new_config}, f, default_flow_style=False)
                
    print(f"Generated {len(configs.datamixes) * len(configs.base_configs) * len(grid_combinations)} configuration files in '{output_dir}' directory.")

    # Make a shell to train CLIP with the configurations we just generated
    # (the list comprehension is voluntarily very specific to show how to select a subset of configurations)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs("logs", exist_ok=True)

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    if config_path:
        main(config_path)