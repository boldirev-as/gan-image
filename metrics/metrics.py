from pytorch_fid import fid_score

from train import load_config
from utils.class_registry import ClassRegistry

metrics_registry = ClassRegistry()


@metrics_registry.add_to_registry(name="fid")
class FID:
    def __call__(self, orig_path, synt_path):
        fid = fid_score.calculate_fid_given_paths(
            paths=['food_data_test/', synt_path],
            batch_size=load_config().data.val_batch_size,
            device=load_config().exp.device,
            dims=2048
        )

        return fid
