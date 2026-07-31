"""S3-backed artifact storage.

The point of this backend is not S3 specifically. It is that a run on a GPU box
and an admin on a laptop can be the same project: with the metastore on Postgres
and artifacts here, ``Model.load()`` works from anywhere that can reach both, and
"train somewhere else" stops being a copying exercise.

Needs boto3::

    pip install mlango[s3]

Works against anything S3-compatible — MinIO, Cloudflare R2, Backblaze B2 — via
``ENDPOINT_URL``. Credentials are boto3's problem, which means the usual
environment variables, instance roles and profiles all work and mlango never
handles a secret it does not have to.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO, Any

from mlango.core.exceptions import BackendNotAvailable, ImproperlyConfigured
from mlango.storage.base import Storage, Written


class S3Storage(Storage):
    """Artifacts in a bucket.

    ``ROOT`` is either a bucket name or an ``s3://bucket/prefix`` URL, so a
    project can share a bucket with everything else it owns.
    """

    def __init__(self, root: str = "", **options: Any):
        super().__init__(root=root, **options)
        if not root:
            raise ImproperlyConfigured(
                "S3Storage needs a bucket: STORAGE = {'BACKEND': "
                "'mlango.storage.s3.S3Storage', 'ROOT': 's3://my-bucket/mlango'}"
            )
        self.bucket, self.prefix = _split(root)
        self.endpoint_url = options.get("endpoint_url") or None
        self.region = options.get("region") or None
        self._client: Any = None

    # -- boto3 ---------------------------------------------------------------

    @property
    def client(self) -> Any:
        """The boto3 client, built on first use.

        Lazily, so importing a settings module that names this backend does not
        require boto3 to be installed — ``manage.py check`` should be able to
        report the missing dependency rather than fail to start.
        """
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise BackendNotAvailable(
                    "S3Storage needs boto3. Install it with: pip install mlango[s3]"
                ) from exc
            self._client = boto3.client(
                "s3", endpoint_url=self.endpoint_url, region_name=self.region
            )
        return self._client

    def key(self, name: str) -> str:
        cleaned = name.replace("\\", "/").lstrip("/")
        if ".." in cleaned.split("/"):
            raise ValueError(f"Refusing to access {name!r}: it escapes the storage prefix.")
        return f"{self.prefix}/{cleaned}" if self.prefix else cleaned

    # -- Storage API ---------------------------------------------------------

    def path(self, name: str) -> str:
        """The ``s3://`` URL for ``name``.

        A URL rather than a filesystem path, because there is no filesystem
        path — anything that needs one goes through :meth:`writable` or
        :meth:`readable`, which is the whole reason those exist.
        """
        return f"s3://{self.bucket}/{self.key(name)}"

    def open(self, name: str, mode: str = "rb") -> IO[Any]:
        import io

        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise ImproperlyConfigured(
                f"S3Storage cannot open {name!r} for writing. Use save_bytes(), or "
                f"writable() for something that needs a real file."
            )
        data = self.read_bytes(name)
        if "b" in mode:
            return io.BytesIO(data)
        return io.StringIO(data.decode("utf-8"))

    def save_bytes(self, name: str, data: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=self.key(name), Body=data)
        return name

    def read_bytes(self, name: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=self.key(name))
        body: bytes = response["Body"].read()
        return body

    def exists(self, name: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.key(name))
            return True
        except Exception:  # noqa: BLE001 - botocore raises a client-specific 404
            return bool(self.listdir(name))

    def delete(self, name: str) -> None:
        keys = [self.key(entry) for entry in self.listdir(name)] or [self.key(name)]
        for batch in (keys[i : i + 1000] for i in range(0, len(keys), 1000)):
            self.client.delete_objects(
                Bucket=self.bucket, Delete={"Objects": [{"Key": key} for key in batch]}
            )

    def size(self, name: str) -> int:
        response = self.client.head_object(Bucket=self.bucket, Key=self.key(name))
        return int(response["ContentLength"])

    def listdir(self, prefix: str = "") -> list[str]:
        under = self.key(prefix) if prefix else self.prefix
        if under and not under.endswith("/"):
            under += "/"
        paginator = self.client.get_paginator("list_objects_v2")
        cut = len(self.prefix) + 1 if self.prefix else 0
        out: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=under):
            for entry in page.get("Contents", []):
                out.append(entry["Key"][cut:])
        return sorted(out)

    # -- staging -------------------------------------------------------------

    @contextmanager
    def writable(self, name: str, *, directory: bool = False) -> Iterator[Written]:
        staging = tempfile.mkdtemp(prefix="mlango-s3-")
        local = os.path.join(staging, os.path.basename(name.rstrip("/")) or "artifact")
        try:
            if directory:
                os.makedirs(local, exist_ok=True)
            yield Written(local, name)
            # Only after the block returns: a failed save must not publish a
            # partial artifact that the next load would happily read.
            if directory:
                self._upload_tree(local, name)
            else:
                with open(local, "rb") as fh:
                    self.save_bytes(name, fh.read())
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @contextmanager
    def readable(self, name: str) -> Iterator[str]:
        if os.path.isabs(name) and os.path.exists(name):
            # A version registered when this project stored artifacts locally.
            yield name
            return

        staging = tempfile.mkdtemp(prefix="mlango-s3-")
        try:
            entries = self.listdir(name)
            if entries == [name.replace("\\", "/").lstrip("/")] or not entries:
                local = os.path.join(staging, os.path.basename(name) or "artifact")
                with open(local, "wb") as fh:
                    fh.write(self.read_bytes(name))
                yield local
            else:
                root = self._download_tree(entries, name, staging)
                yield root
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def locate(self, name: str) -> str:
        if os.path.isabs(name) and os.path.exists(name):
            return name
        raise ImproperlyConfigured(
            f"{name!r} lives in S3 and has no local path. Read it with "
            f"storage.readable(name), which downloads it for the length of a with-block, "
            f"or storage.fetch(name), which caches a copy."
        )

    def fetch(self, name: str) -> str:
        """Download once into a cache directory and return the local path.

        Content is addressed by name and never rewritten in place — a
        materialised dataset version is immutable by construction — so a cached
        copy cannot go stale, and re-reading one costs nothing after the first
        time.
        """
        if os.path.isabs(name) and os.path.exists(name):
            return name

        cache = os.path.join(tempfile.gettempdir(), "mlango-s3-cache", self.bucket)
        local = os.path.join(cache, *self.key(name).split("/"))
        if os.path.exists(local):
            return local

        os.makedirs(os.path.dirname(local), exist_ok=True)
        partial = local + ".part"
        with open(partial, "wb") as fh:
            fh.write(self.read_bytes(name))
        os.replace(partial, local)
        return local

    # -- helpers -------------------------------------------------------------

    def _upload_tree(self, local_root: str, name: str) -> None:
        for dirpath, _dirnames, filenames in os.walk(local_root):
            for filename in filenames:
                full = os.path.join(dirpath, filename)
                relative = os.path.relpath(full, local_root).replace("\\", "/")
                with open(full, "rb") as fh:
                    self.save_bytes(f"{name.rstrip('/')}/{relative}", fh.read())

    def _download_tree(self, entries: list[str], name: str, staging: str) -> str:
        base = name.replace("\\", "/").strip("/")
        root = os.path.join(staging, os.path.basename(base) or "artifact")
        for entry in entries:
            relative = entry[len(base) :].lstrip("/") if entry.startswith(base) else entry
            target = os.path.join(root, *relative.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as fh:
                fh.write(self.read_bytes(entry))
        return root

    def __repr__(self) -> str:
        return f"<S3Storage bucket={self.bucket!r} prefix={self.prefix!r}>"


def _split(root: str) -> tuple[str, str]:
    """``s3://bucket/prefix`` or ``bucket/prefix`` into its two halves."""
    cleaned = root[len("s3://") :] if root.startswith("s3://") else root
    bucket, _, prefix = cleaned.strip("/").partition("/")
    return bucket, prefix.strip("/")


__all__ = ["S3Storage"]
