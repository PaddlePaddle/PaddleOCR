import os
from .base_logger import BaseLogger
from typing import Dict, Optional

class MlflowLogger(BaseLogger):
    def __init__(self,
                save_dir: str, 
                exp_name: Optional[str] = None,
                run_name: Optional[str] = None,
                tags: Optional[dict] = None,
                params: Optional[dict] = None,
                tracking_uri: Optional[str] = None,
                **kwargs):

        self.import_mlflow()
        self.save_dir = save_dir
        self.exp_name = exp_name
        self.run_name = run_name
        self.tags = tags
        self.params = params
        self.tracking_uri = tracking_uri

        if self.tracking_uri is not None:
            self.mlflow.set_tracking_uri(self.tracking_uri)
        else:
            mlflow_writer_path = "{}/mlflow".format(self.save_dir)
            file_url = os.path.abspath(mlflow_writer_path)
            self.mlflow.set_tracking_uri(file_url)

        if self.mlflow.get_experiment_by_name(self.exp_name) is None:
            self.mlflow.create_experiment(self.exp_name)

        self.mlflow.set_experiment(self.exp_name)

        self._run = None
        _ = self.run

        if self.run_name is not None:
            self._run = self.mlflow.start_run(run_name=self.run_name)
            self.mlflow.set_tag('mlflow.runName', self.run_name)
        if self.tags is not None:
            self.mlflow.set_tags(self.tags)
        if self.params is not None:
            self.mlflow.log_params(self.params)


    def import_mlflow(self) -> None:
        try:
            import mlflow
            import mlflow.paddle as mlflow_paddle
        except ImportError:
            raise ImportError(
                'Please run "pip install mlflow" to install mlflow')
        self.mlflow = mlflow
        self.mlflow_paddle = mlflow_paddle

    @property
    def run(self):
        return

    def log_metrics(self, metrics, prefix=None, step=None):
        if not prefix:
            prefix = ""
        updated_metrics = {prefix.lower() + "/" + k: v for k, v in metrics.items()}
        self.mlflow.log_metrics(updated_metrics, step=step)

    def log_model(self, is_best, prefix, metadata=None):
        model_path = os.path.join(self.save_dir, prefix + '.pdparams')
        config_path = os.path.join(self.save_dir, "config.yml")
        if is_best:
            artifact_path = '{}'.format(prefix)
        else:
            artifact_path = '{}'.format(prefix)

        self.mlflow.log_artifact(model_path, artifact_path=artifact_path)
        self.mlflow.log_artifact(config_path, artifact_path=artifact_path)

        if metadata is not None:
            metadata_path = os.path.join(artifact_path, "epoch_" + str(metadata["best_epoch"]) + ".json")
            self.mlflow.log_dict(metadata, artifact_file=metadata_path)

    def close(self):
        self.mlflow.end_run()
        return
