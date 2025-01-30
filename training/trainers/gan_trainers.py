import numpy as np
import torch
from PIL import Image

from utils.class_registry import ClassRegistry
from utils.model_utils import requires_grad
from training.trainers.base_trainer import BaseTrainer

from models.gan_models import gens_registry, discs_registry, VerySimpleGenarator, VerySimpleDiscriminator
from training.optimizers import optimizers_registry, Adam_
from training.losses.gan_losses import GANLossBuilder, gen_losses_registry, disc_losses_registry

gan_trainers_registry = ClassRegistry()


def get_grad_norm(model):
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
    return total_norm ** 0.5


@gan_trainers_registry.add_to_registry(name="base_gan_trainer")
class BaseGANTrainer(BaseTrainer):
    def setup_models(self):
        self.dicriminator = discs_registry['base_disc'](self.config.discriminator_args)
        self.dicriminator.to(self.config.exp.device)
        self.generator = gens_registry['base_gen'](self.config.generator_args)
        self.generator.to(self.config.exp.device)

    def setup_optimizers(self):
        self.generator_optimizer = optimizers_registry['adam'](self.generator.parameters(),
                                                               **self.config.gen_optimizer_args)
        self.dicriminator_optimizer = optimizers_registry['adam'](self.dicriminator.parameters(), lr=5e-6,
                                                                  betas=(0.5, 0.999))

    def setup_losses(self):
        self.loss_builder = GANLossBuilder(self.config)

    def to_train(self):
        self.dicriminator.train()
        self.generator.train()

    def to_eval(self):
        self.dicriminator.eval()
        self.generator.eval()

    def train_step(self):
        batch = next(self.train_dataloader)
        batch_size = batch['images'].size(0)

        real_images = batch['images'].to(self.config.exp.device)

        # Добавляем шум, чтобы дискриминатор не переобучался
        real_images += 0.1 * torch.randn_like(real_images)

        z = torch.randn((batch_size, self.config.generator_args.z_dim), device=self.config.exp.device)
        generated_images = self.generator(z)

        # Обучаем дискриминатор
        batch['real_preds'] = self.dicriminator(real_images)
        batch['fake_preds'] = self.dicriminator(generated_images.detach())  # detach, чтобы генератор не обновлялся

        d_loss, loss_dict_disc = self.loss_builder.calculate_loss(batch, "disc")

        self.dicriminator_optimizer.zero_grad()
        d_loss.backward()

        d_grad_norm = get_grad_norm(self.dicriminator)

        torch.nn.utils.clip_grad_norm_(self.dicriminator.parameters(), max_norm=1.0)  # Градиентный клиппинг
        self.dicriminator_optimizer.step()

        batch['fake_preds'] = self.dicriminator(generated_images)  # Без detach()
        g_loss, loss_dict_gen = self.loss_builder.calculate_loss(batch, "gen")

        self.generator_optimizer.zero_grad()
        g_loss.backward()

        g_grad_norm = get_grad_norm(self.generator)

        self.generator_optimizer.step()

        return {'d_loss': d_loss.item(), 'g_loss': g_loss.item(), 'd_grad_norm': d_grad_norm,
                'g_grad_norm': g_grad_norm}

    def save_checkpoint(self):
        state = {
            "discriminator_state_dict": self.dicriminator.state_dict(),
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_optimizer_state_dict": self.dicriminator_optimizer.state_dict(),
            "generator_optimizer_state_dict": self.generator_optimizer.state_dict(),
        }
        torch.save(state, f"{self.experiment_dir}/checkpoint.pth.tar")

    def synthesize_images(self):
        # TO DO
        # synthesize images and save to self.experiment_dir/images
        # synthesized additional batch of images to log

        import os

        os.makedirs(f"{self.experiment_dir}/images", exist_ok=True)

        self.to_eval()
        batch_size = self.config.data.train_batch_size

        total_images = []

        for _ in range(64):
            z = torch.randn((batch_size, self.config.generator_args.z_dim), device=self.config.exp.device)
            with torch.no_grad():
                batch_of_images = self.generator(z)

            for img in batch_of_images:
                img = img.cpu().detach().numpy()
                img = (img * 0.5 + 0.5) * 255
                img = img.astype(np.uint8)
                img = np.transpose(img, (1, 2, 0))
                img = Image.fromarray(img)

                total_images.append(img)

        for i, img in enumerate(total_images):
            img_path = f"{self.experiment_dir}/images/synthesized_image_{i}.png"
            img.save(img_path)

        return total_images[:10], f"{self.experiment_dir}/images"
