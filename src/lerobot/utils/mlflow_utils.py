#!/usr/bin/env python

import logging
import json
import math
import os
from pathlib import Path
from typing import Any

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import PRETRAINED_MODEL_DIR

MLFLOW_RUN_ID_FNAME = "mlflow_run_id.txt"


def cfg_to_group(cfg: TrainPipelineConfig) -> str:
    parts: list[str] = [f"policy:{cfg.policy.type}"]
    if cfg.seed is not None:
        parts.append(f"seed:{cfg.seed}")
    if cfg.dataset is not None:
        parts.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        parts.append(f"env:{cfg.env.type}")
    return "-".join(parts)


def cfg_to_tags(cfg: TrainPipelineConfig) -> dict[str, str]:
    tags = {
        "policy": cfg.policy.type,
    }
    if cfg.seed is not None:
        tags["seed"] = str(cfg.seed)
    if cfg.dataset is not None:
        tags["dataset"] = str(cfg.dataset.repo_id)
    if cfg.env is not None:
        tags["environment"] = cfg.env.type
    return tags


def _load_run_id_from_disk(log_dir: Path) -> str | None:
    run_id_path = log_dir / MLFLOW_RUN_ID_FNAME
    if not run_id_path.exists():
        return None
    return run_id_path.read_text(encoding="utf-8").strip()


def _store_run_id_to_disk(log_dir: Path, run_id: str) -> None:
    run_id_path = log_dir / MLFLOW_RUN_ID_FNAME
    run_id_path.parent.mkdir(parents=True, exist_ok=True)
    run_id_path.write_text(run_id, encoding="utf-8")


class MlflowLogger:
    """Helper to log training artifacts and metrics to MLflow."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.mlflow
        self.log_dir = Path(cfg.output_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)
        self._warned_keys: set[str] = set()

        os.environ.setdefault("MLFLOW_DISABLE_ENV_VARS", "1")
        import mlflow

        if self.log_dir is None:
            raise ValueError("cfg.output_dir must be set before initializing MlflowLogger")

        tracking_uri = self.cfg.tracking_uri
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            local_store = self.log_dir / "mlruns"
            local_store.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"file://{local_store.resolve()}")

        experiment = self.cfg.experiment or "lerobot"
        mlflow.set_experiment(experiment)

        run_id = self.cfg.run_id
        if cfg.resume and not run_id:
            run_id = _load_run_id_from_disk(self.log_dir)

        run = mlflow.start_run(run_id=run_id, run_name=self.job_name)
        self._mlflow = mlflow
        self._active_run = run

        cfg.mlflow.run_id = run.info.run_id
        _store_run_id_to_disk(self.log_dir, run.info.run_id)

        tags = cfg_to_tags(cfg)
        tags["group"] = self._group
        if self.cfg.description:
            tags["description"] = self.cfg.description
        if self.cfg.tags:
            tags.update(self.cfg.tags)
        mlflow.set_tags(tags)

        config_path = self.log_dir / "mlflow_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as fp:
            json.dump(cfg.to_dict(), fp, indent=2, default=str)
        mlflow.log_artifact(str(config_path), artifact_path="config")
        print(colored("Logs will be synced with MLflow.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(run.info.run_id, 'yellow', attrs=['bold'])}")

    def log_policy(self, checkpoint_dir: Path) -> None:
        if self.cfg.disable_artifact:
            return
        if not checkpoint_dir.exists():
            logging.warning("Checkpoint directory %s does not exist, skipping MLflow artifact upload", checkpoint_dir)
            return
        artifact_path = f"checkpoints/{checkpoint_dir.name}"
        model_file = checkpoint_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE
        if model_file.exists():
            self._mlflow.log_artifact(str(model_file), artifact_path=artifact_path)
        else:
            logging.warning(
                "Model file %s is missing; skipping MLflow artifact upload for checkpoint %s",
                model_file,
                checkpoint_dir,
            )

    def log_dict(
        self,
        d: dict[str, Any],
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        step_value = step
        metrics: dict[str, float] = {}
        if custom_step_key is not None:
            if custom_step_key not in d:
                raise KeyError(f"custom_step_key '{custom_step_key}' not found in logging dict")
            step_val_raw = d[custom_step_key]
            if not isinstance(step_val_raw, (int, float)):
                raise TypeError("custom_step_key values must be numeric to be used as MLflow steps")
            step_value = int(step_val_raw)
            metrics[f"{mode}/{custom_step_key}"] = float(step_val_raw)

        for key, value in d.items():
            if key == custom_step_key:
                continue
            if isinstance(value, (int, float)) and math.isfinite(value):
                metrics[f"{mode}/{key}"] = float(value)
                continue
            if isinstance(value, str):
                if key not in self._warned_keys:
                    logging.warning(
                        "MLflow logging of key '%s' is skipped because string values are not supported as metrics.",
                        key,
                    )
                    self._warned_keys.add(key)
                continue
            if key not in self._warned_keys:
                logging.warning(
                    "MLflow logging of key '%s' with type '%s' is not supported and will be skipped.",
                    key,
                    type(value),
                )
                self._warned_keys.add(key)

        if metrics:
            self._mlflow.log_metrics(metrics, step=step_value)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        path_obj = Path(video_path)
        if not path_obj.exists():
            logging.warning("Video path %s does not exist; skipping MLflow upload", video_path)
            return
        artifact_subdir = f"{mode}/videos/step_{step}"
        self._mlflow.log_artifact(str(path_obj), artifact_path=artifact_subdir)
