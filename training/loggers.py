import wandb
import torch
from PIL import Image
from collections import defaultdict
import os
import numpy as np

from dotenv import load_dotenv

load_dotenv()


class WandbLogger:
    def __init__(self, config):
        wandb.login(key=os.environ['WANDB_KEY'].strip())

        # Проверяем, есть ли чекпойнт для возобновления
        if config.train.checkpoint_path is not None:
            self.wandb_args = {
                "id": wandb.util.generate_id(),
                "project": config.exp.project,
                "name": config.exp.name,
                "config": config,
                "resume": "must",
            }
        else:
            self.wandb_args = {
                "id": wandb.util.generate_id(),
                "project": config.exp.project,
                "name": config.exp.name,
                "config": config,
            }

        wandb.init(**self.wandb_args, resume="allow")

    @staticmethod
    def log_values(values_dict: dict, step: int):
        """
        Логирует числовые значения (например, loss, метрики) в WandB.
        :param values_dict: словарь {метрика: значение}
        :param step: текущий шаг обучения
        """
        wandb.log(values_dict, step=step)

    @staticmethod
    def log_images(images: dict, step: int):
        """
        Логирует изображения в WandB.
        :param images: словарь {имя: изображение (PIL.Image или torch.Tensor)}
        :param step: текущий шаг обучения
        """
        logs = {}
        for name, img in images.items():
            if isinstance(img, torch.Tensor):  # Преобразуем тензор в изображение
                img = img.cpu().detach().numpy()
                img = (img * 0.5 + 0.5) * 255
                img = img.astype(np.uint8)
                img = np.transpose(img, (1, 2, 0))
                img = Image.fromarray(img)
            elif isinstance(img, np.ndarray):  # Если это NumPy-массив
                img = Image.fromarray(((img + 0.5) * 255).astype(np.uint8))
            logs[name] = wandb.Image(img)

        wandb.log(logs, step=step)


class TrainingLogger:
    def __init__(self, config):
        self.logger = WandbLogger(config)
        self.losses_memory = defaultdict(list)

    def log_train_losses(self, step: int):
        """
        Усредняет накопленные лоссы и логирует их, затем очищает память.
        :param step: текущий шаг
        """
        if not self.losses_memory:
            return

        avg_losses = {loss_name: sum(vals) / len(vals) for loss_name, vals in self.losses_memory.items()}
        self.logger.log_values(avg_losses, step)

        # Очищаем память
        self.losses_memory.clear()

    def log_val_metrics(self, val_metrics: dict, step: int):
        """
        Логирует метрики валидации.
        :param val_metrics: словарь {метрика: значение}
        :param step: текущий шаг
        """
        self.logger.log_values(val_metrics, step)

    def log_batch_of_images(self, batch: torch.Tensor, step: int, images_type: str = ""):
        """
        Логирует батч изображений.
        :param batch: Тензор изображений [B, C, H, W]
        :param step: текущий шаг
        :param images_type: текстовое описание (например, "real", "generated")
        """
        images = {f"{images_type}_{i}": batch[i] for i in range(min(8, batch.shape[0]))}  # Логируем 8 примеров
        self.logger.log_images(images, step)

    def update_losses(self, losses_dict):
        """
        Добавляет текущие лоссы в память для усреднения.
        :param losses_dict: словарь {имя лосса: значение}
        """
        for loss_name, loss_val in losses_dict.items():
            self.losses_memory[loss_name].append(loss_val)
