import itertools
import yaml
import os

# Datamixes, sets basic info about each dataset
# The weight is determined relative of the size of each dataset
# (so that each dataset is represented in the mixture according to its initial size)

# You can define several data mixes, to do the training with different mixtures of datasets for different aims.

base_folder = "/"

datamixes = {
    "combined_dataset_MRI": f"""dataset_configs:
  - combined_dataset_MRI:
      dataset_name: "{os.path.join(base_folder, 'IRM_jsonl/IRM-train.jsonl')}"
      image_column: "modalities"
      caption_column: "text"
      weight: 1"""
}

# Base configurations
base_configs = {
    "initial": {
        "learning_rate": 5.0e-4,
        "warmup_steps": 2000,
        "lr_scheduler_type": "cosine",
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "adam_epsilon": 1.0e-6,
        "weight_decay": 0.2,
        "num_train_epochs": 32
    },
    "fine_tuning": {
        "learning_rate": 1.0e-4,
        "warmup_steps": 1000,
        "lr_scheduler_type": "linear",
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "weight_decay": 0.1,
        "num_train_epochs": 20
    },
    "aggressive_training": {
        "learning_rate": 1.0e-3,
        "warmup_steps": 500,
        "lr_scheduler_type": "cosine",
        "adam_beta1": 0.85,
        "adam_beta2": 0.95,
        "weight_decay": 0.3,
        "num_train_epochs": 40
    },
    "regularization_focused": {
        "learning_rate": 2.5e-4,
        "warmup_steps": 2000,
        "lr_scheduler_type": "cosine",
        "adam_beta1": 0.95,
        "adam_beta2": 0.999,
        "weight_decay": 0.4,
        "num_train_epochs": 32
    }
}

# Hyperparameter ranges for grid search
param_ranges = {
    "learning_rate": [1.0e-4, 5.0e-4, 1.0e-3],
    "num_train_epochs": [40]
}

def main():
    # Create output directory
    output_dir = "configurations"
    os.makedirs(output_dir, exist_ok=True)

    # Generate configurations
    for datamix_name, datamix in datamixes.items():
        for config_name, base_config in base_configs.items():
            # Create grid search combinations for specified parameters
            param_names = param_ranges.keys()
            param_values = param_ranges.values()
            grid_combinations = list(itertools.product(*param_values))
            
            for idx, combination in enumerate(grid_combinations):
                new_config = base_config.copy()
                new_config.update(dict(zip(param_names, combination)))
                
                # Save each configuration as a YAML file
                config_filename = f"{datamix_name}_{config_name}_config_{idx + 1}"
                config_filepath = os.path.join(output_dir, config_filename)
                
                # Basic configuration, common to all
                begin = f"""output_dir: "./models/{config_filename}"
vision_model_name: "openai/clip-vit-base-patch32"
text_model_name: "naver/splade-v3"
remove_unused_columns: false
do_train: true
per_device_train_batch_size: 64
dataloader_drop_last: true
overwrite_output_dir: true
save_steps: 150
"""

                with open(config_filepath+".yaml", "w") as f:
                    f.write(begin)
                    f.write(datamix+"\n")
                    yaml.dump(new_config, f, default_flow_style=False)
                
    print(f"Generated {len(datamixes) * len(base_configs) * len(grid_combinations)} configuration files in '{output_dir}' directory.")

    # Make a shell to train CLIP with the configurations we just generated
    # (the list comprehension is voluntarily very specific to show how to select a subset of configurations)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs("logs", exist_ok=True)

    with open("../train_all.sh", "w") as f:
        f.write("\n".join(f"accelerate launch train.py {current_dir}/configurations/{config_file} --multi_gpu --num_processes 4 &> {current_dir}/logs/log_{config_file[:-5]}.txt" for config_file in sorted(os.listdir("configurations")) if not os.path.exists(f"logs/log_{config_file[:-5]}.txt") and "xr" in config_file and "2" in config_file) + "\n")

if __name__ == "__main__":
    main()