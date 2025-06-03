"""
This file contains configuration management implementations for GRPO training
"""
import yaml
import argparse
import os
from typing import Dict, Any

def load_config(config_path: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load configuration from YAML file with optional overrides"""
    
    # load base config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # apply command line overrides (if any)
    if overrides:
        config = deep_update(config, overrides)
    
    return config

def deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Recursively update nested dictionaries"""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict:
            base_dict[key] = deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def parse_override_args():
    """Parse command line arguments for config overrides"""
    parser = argparse.ArgumentParser(description="GRPO Training with configurable parameters")
    
    # required args
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to YAML configuration file"
    )
    
    # override args
    parser.add_argument("--experiment_name", type=str, help="Override experiment name")
    parser.add_argument("--model_path", type=str, help="Override model path")
    parser.add_argument("--dataset_size", type=int, help="Override dataset size")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--num_generations", type=int, help="Override number of generations")
    parser.add_argument("--temperature", type=float, help="Override temperature")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging")
    
    args = parser.parse_args()
    
    # build overrides dict
    overrides = {}
    if args.experiment_name:
        overrides["experiment"] = {"name": args.experiment_name}
    if args.model_path:
        overrides["model"] = {"name_or_path": args.model_path}
    if args.dataset_size:
        overrides["data"] = {"dataset_size": args.dataset_size}
    if args.batch_size:
        overrides["training"] = {"per_device_train_batch_size": args.batch_size}
    if args.num_generations:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["num_generations"] = args.num_generations
    if args.temperature:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["temperature"] = args.temperature
    if args.output_dir:
        overrides["experiment"] = overrides.get("experiment", {})
        overrides["experiment"]["output_dir"] = args.output_dir
    if args.disable_wandb:
        overrides["wandb"] = {"enabled": False}
    
    return args.config, overrides

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters"""
    required_sections = ["experiment", "model", "data", "training", "reward"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # validate required parameters
    if config["training"]["per_device_train_batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    
    if config["training"]["num_generations"] <= 0:
        raise ValueError("num_generations must be positive")
    
    # check file paths exist
    for file_path in config["data"]["training_files"]:
        if not os.path.exists(file_path):
            raise ValueError(f"Training file not found: {file_path}")
