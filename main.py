from torch.utils.data import DataLoader
from src.layers.Models import TinyDiffusion
from src.data.BaseData import DataSetTinyDiffusion
import hydra
from omegaconf import DictConfig
from src.utils.manage_device import select_device
from src.utils.manage_data import manage_data_names


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print(cfg)
    device = select_device(cfg.device)

    model_name = cfg.model_type.model_name

    # Correct way to merge dictionaries and add a new key in Python
    model_config = {**cfg.model, "device": device}
    dataset_config = {**cfg.dataset, "device": device}
    dataset_config["name_dataset"] = [("dino", ""), ("moons", "")]
    trainer_config = {
        **cfg.trainer,
        "data_name": dataset_config["name_dataset"],
        "device": device,
    }
    optimizer_config = {**cfg.optimizer}
    diff_config = {**cfg.Diffusion, "device": device}

    print(
        f"""
         device: {device}
         model_name: {model_name}
         model configuration: {model_config}
         trainer configuration: {trainer_config}
         optimizer configuration: {optimizer_config}
         dataset configuration: {dataset_config}
         diffusion configuration : {diff_config}
        """
    )
    tinyDiff_builder = TinyDiffusion(
        model_name, model_config, trainer_config, optimizer_config, diff_config, device
    )
    print(tinyDiff_builder)
    dataset = DataSetTinyDiffusion(**dataset_config)
    print(dataset)
    train_loader = DataLoader(
        dataset, batch_size=int(trainer_config["batch_size"]), shuffle=True
    )
    print(train_loader)
    tinyDiff_builder.train(train_loader)


if __name__ == "__main__":
    main()
