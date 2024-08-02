import os
from .base_logger import BaseLogger
from ppocr.utils.logging import get_logger


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project=None,
        name=None,
        id=None,
        entity=None,
        save_dir=None,
        config=None,
        **kwargs,
    ):
        try:
            import wandb

            self.wandb = wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install wandb using `pip install wandb`")

        self.project = project
        self.name = name
        self.id = id
        self.save_dir = save_dir
        self.config = config
        self.kwargs = kwargs
        self.entity = entity
        self._run = None
        self._wandb_init = dict(
            project=self.project,
            name=self.name,
            id=self.id,
            entity=self.entity,
            dir=self.save_dir,
            resume="allow",
        )
        self._wandb_init.update(**kwargs)
        self.logger = get_logger()

        _ = self.run

        if self.config:
            self.run.config.update(self.config)

    @property
    def run(self):
        if self._run is None:
            if self.wandb.run is not None:
                self.logger.info(
                    "There is a wandb run already in progress "
                    "and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()`"
                    "before instantiating `WandbLogger`."
                )
                self._run = self.wandb.run
            else:
                self._run = self.wandb.init(**self._wandb_init)
        return self._run

    def log_metrics(self, metrics, prefix=None, step=None):
        if not prefix:
            prefix = ""
        updated_metrics = {prefix.lower() + "/" + k: v for k, v in metrics.items()}

        self.run.log(updated_metrics, step=step)

    def log_model(self, is_best, prefix, metadata=None):
        model_path = os.path.join(self.save_dir, prefix + ".pdparams")
        artifact = self.wandb.Artifact(
            "model-{}".format(self.run.id), type="model", metadata=metadata
        )
        artifact.add_file(model_path, name="model_ckpt.pdparams")

        aliases = [prefix]
        if is_best:
            aliases.append("best")

        self.run.log_artifact(artifact, aliases=aliases)

    def close(self):
        self.run.finish()
