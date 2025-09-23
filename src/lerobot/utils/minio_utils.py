#!/usr/bin/env python

"""Utilities to push checkpoints to a MinIO/S3 compatible object store."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable

from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from lerobot.configs.train import TrainPipelineConfig


class MinioUploader:
    """Upload Lerobot training artifacts to MinIO."""

    def __init__(self, cfg: TrainPipelineConfig):
        try:
            import boto3  # pylint: disable=import-error
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "boto3 is required to enable MinIO uploads. Install it with `pip install boto3`."
            ) from exc

        self._cfg = cfg
        minio_cfg = cfg.minio

        # Normalise endpoint URL
        endpoint = minio_cfg.endpoint or ""
        if endpoint and not endpoint.startswith("http"):
            scheme = "https" if minio_cfg.secure else "http"
            endpoint = f"{scheme}://{endpoint}"

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint or None,
            aws_access_key_id=minio_cfg.access_key,
            aws_secret_access_key=minio_cfg.secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name=minio_cfg.region,
        )
        self._bucket = minio_cfg.bucket
        self._upload_history = minio_cfg.upload_history
        self._keep_local_copy = minio_cfg.keep_local_copy
        prefix_parts = [part for part in (minio_cfg.prefix, cfg.job_name) if part]
        if prefix_parts:
            self._base_prefix = "/".join(prefix_parts)
        else:
            # Fallback to directory name to avoid collisions when job_name is missing
            self._base_prefix = Path(cfg.output_dir).name

        self._ensure_bucket(minio_cfg.create_bucket)
        logging.info("MinIO uploads enabled. Bucket=%s Prefix=%s", self._bucket, self._base_prefix)

    # ------------------------------------------------------------------
    def upload_checkpoint(self, checkpoint_dir: Path, aliases: Iterable[str] | None = None) -> None:
        """Upload a checkpoint directory to MinIO.

        Args:
            checkpoint_dir: Local directory produced by `save_checkpoint`.
            aliases: Optional labels (e.g. ["last"], ["best"]) that should point to the
                same checkpoint contents under `checkpoints/<alias>/`.
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            logging.warning("MinIO upload skipped, checkpoint directory missing: %s", checkpoint_dir)
            return

        aliases = list(aliases or [])
        step_prefix = self._remote_prefix("checkpoints", checkpoint_dir.name)
        if self._upload_history:
            self._upload_directory(checkpoint_dir, step_prefix)

        for alias in aliases:
            alias_prefix = self._remote_prefix("checkpoints", alias)
            self._delete_prefix(alias_prefix)
            self._upload_directory(checkpoint_dir, alias_prefix)

        if not self._keep_local_copy:
            self._remove_local(checkpoint_dir)

    # ------------------------------------------------------------------
    def _remote_prefix(self, *parts: str) -> str:
        joined = "/".join(part.strip("/") for part in parts if part)
        if self._base_prefix:
            return f"{self._base_prefix}/{joined}" if joined else self._base_prefix
        return joined

    def _ensure_bucket(self, create_bucket: bool) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in {"404", "NoSuchBucket"}:
                if create_bucket:
                    logging.info("Creating MinIO bucket %s", self._bucket)
                    self._client.create_bucket(Bucket=self._bucket)
                else:
                    raise ValueError(
                        f"MinIO bucket '{self._bucket}' does not exist. "
                        "Set minio.create_bucket=true to create it automatically."
                    ) from exc
            else:
                raise

    def _upload_directory(self, local_dir: Path, remote_prefix: str) -> None:
        for file_path in local_dir.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(local_dir).as_posix()
            key = f"{remote_prefix}/{relative}" if remote_prefix else relative
            logging.debug("Uploading %s -> s3://%s/%s", file_path, self._bucket, key)
            self._client.upload_file(str(file_path), self._bucket, key)

    def _delete_prefix(self, remote_prefix: str) -> None:
        if not remote_prefix:
            return
        paginator = self._client.get_paginator("list_objects_v2")
        delete_items = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=remote_prefix):
            contents = page.get("Contents") or []
            for obj in contents:
                delete_items.append({"Key": obj["Key"]})
            if delete_items:
                self._client.delete_objects(Bucket=self._bucket, Delete={"Objects": delete_items})
                delete_items.clear()

    def _remove_local(self, checkpoint_dir: Path) -> None:
        try:
            shutil.rmtree(checkpoint_dir)
        except OSError as exc:
            logging.warning("Failed to remove local checkpoint %s: %s", checkpoint_dir, exc)
