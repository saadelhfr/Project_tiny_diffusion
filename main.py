import hydra
import torch
from src.layers.Models import TinyDiffusion
from src.utils.manage_device import select_device


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg):
    print(cfg)
    device = select_device(cfg["device"])

    model_name = cfg["model_type"]["model_name"]
    model_config = cfg["model"]
    trainer_config = cfg["trainer"]
    optimizer_config = cfg["optimizer"]
    print(
        f"""
         device : {device}
         model_name  : {model_name}
         model configuration :  {model_config}
         trainer configuration : {trainer_config}
         optimizer configuration : {optimizer_config}

        """
    )


if __name__ == "__main__":
    main()
