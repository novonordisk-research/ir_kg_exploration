from typing import Union
from pathlib import Path

import contextlib
import hashlib
import warnings
import shutil

import boto3
import boto3.s3
import boto3.s3.transfer


def _remove_prefix(s, prefix):
    if s.startswith(prefix):
        s = s[len(prefix) :]
    return s


DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


def calculate_s3_etag(file_path, chunk_size=DEFAULT_CHUNK_SIZE):
    md5s = []

    with open(file_path, "rb") as fp:
        while True:
            data = fp.read(chunk_size)
            if not data:
                break
            md5s.append(hashlib.md5(data))

    if len(md5s) < 1:
        return '"{}"'.format(hashlib.md5().hexdigest())

    if len(md5s) == 1:
        return '"{}"'.format(md5s[0].hexdigest())

    digests = b"".join(m.digest() for m in md5s)
    digests_md5 = hashlib.md5(digests)
    return '"{}-{}"'.format(digests_md5.hexdigest(), len(md5s))


class S3DataStoreFile:
    def __init__(self, s3_data_store: "S3DataStore", file_name: str):
        self.s3_data_store = s3_data_store
        self.file_name = file_name
        self.key = self.s3_data_store.key_from_file_name(self.file_name)
        self.bucket_name = self.s3_data_store.bucket_name
        self.prefix = self.s3_data_store.prefix
        self.local_path = self.s3_data_store._local_path(self.file_name)

    def __hash__(self) -> int:
        return hash(self.file_name)

    def __eq__(self, other: "S3DataStoreFile") -> bool:
        return self.file_name == other.file_name

    def __ne__(self, other: "S3DataStoreFile"):
        return not self.__eq__(other)

    def download(self):
        return self.s3_data_store.download_file(self.file_name)

    def upload(self, overwrite=False):
        return self.s3_data_store.upload_file(
            file_name=self.file_name, overwrite=overwrite
        )

    def get_file(self, force_download=False):
        return self.s3_data_store.get_file(
            file_name=self.file_name, force_download=force_download
        )

    def has_local_copy(self):
        return self.s3_data_store.file_has_local_copy(self.file_name)

    def has_remote_copy(self):
        return self.s3_data_store.file_has_remote_copy(self.file_name)

    def delete(self, delete_remote=False):
        return self.s3_data_store.delete_file(
            self.file_name, delete_remote=delete_remote
        )

    def has_matching_etags(self):
        return self.s3_data_store.file_has_matching_etags(self.file_name)

    def get_remote_etag(self):
        return self.s3_data_store.get_remote_etag(self.file_name)

    def get_local_etag(self):
        return self.s3_data_store.get_local_etag(self.file_name)

    def __str__(self) -> str:
        return f"<S3DataStoreFile: {self.bucket_name}/{self.key}>"

    def __repr__(self) -> str:
        return f"<S3DataStoreFile: {self.bucket_name}/{self.key}>"


DEFAULT_TRANSFER_CONFIG = boto3.s3.transfer.TransferConfig(
    io_chunksize=DEFAULT_CHUNK_SIZE
)


class S3DataStore:
    def __init__(
        self,
        aws_access_key_id,
        aws_secret_access_key,
        aws_session_token,
        bucket_name,
        prefix,
        local_directory: Union[str, Path],
        s3_transfer_config=DEFAULT_TRANSFER_CONFIG,
    ):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.bucket_name = bucket_name
        self.prefix = prefix
        if not self.prefix.endswith("/"):
            warnings.warn('Prefix does not end with "/". Is this intentional?')
        self.local_directory = Path(local_directory)
        self.local_directory.mkdir(parents=True, exist_ok=True)
        n_files = len(list(self.local_directory.rglob("*")))
        print(f'Found {n_files} files in local directory "{self.local_directory}".')

        self.s3_transfer_config = s3_transfer_config

        self.s3 = boto3.resource(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )
        self.bucket = self.s3.Bucket(bucket_name)

    def list_files(self, prefix=""):
        remote_files = self.list_remote_files(prefix=prefix)
        local_files = self.list_local_files(prefix=prefix)

        return list(set([*remote_files, *local_files]))

    def list_local_files(self, prefix=""):
        local_file_paths = self._list_local_files()
        local_file_names = [
            f.resolve().relative_to(self.local_directory.resolve())
            for f in local_file_paths
        ]
        local_file_names = [f for f in local_file_names]
        return [
            S3DataStoreFile(s3_data_store=self, file_name=str(f))
            for f in local_file_names
            if str(f).startswith(prefix)
        ]

    def list_remote_files(self, prefix=""):
        r = []
        prefix = self.prefix + prefix
        for o in self.bucket.objects.filter(Prefix=prefix):
            r.append(S3DataStoreFile(self, o.key.replace(self.prefix, "")))
        return r

    def download_file(self, file_name: Union[str, "S3DataStoreFile"]):
        if isinstance(file_name, S3DataStoreFile):
            file_name = file_name.file_name
        key = self.key_from_file_name(file_name)

        local_path = self._local_path(file_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.bucket.download_file(
            Key=key, Filename=local_path, Config=self.s3_transfer_config
        )
        return S3DataStoreFile(s3_data_store=self, file_name=file_name)

    def upload_file(self, file_name: Union[str, "S3DataStoreFile"], overwrite=False):
        if isinstance(file_name, S3DataStoreFile):
            file_name = file_name.file_name
        if self.file_has_remote_copy(file_name) and not overwrite:
            msg = f'File "{file_name}" already exists on remote. Run with overwrite=True to overwrite.'
            raise ValueError(msg)
        key = self.key_from_file_name(file_name)
        local_file_path = self._local_path(file_name)
        self.bucket.upload_file(
            Filename=local_file_path, Key=key, Config=self.s3_transfer_config
        )
        return S3DataStoreFile(s3_data_store=self, file_name=file_name)

    def delete_file(
        self, file_name: Union[str, "S3DataStoreFile"], delete_remote=False
    ):
        if isinstance(file_name, S3DataStoreFile):
            file_name = file_name.file_name
        if self.file_has_local_copy(file_name=file_name):
            self._local_path(file_name=file_name).unlink()
        if delete_remote and self.file_has_remote_copy(file_name=file_name):
            self.bucket.Object(key=self.key_from_file_name(file_name)).delete()

    def get_file(
        self,
        file_name: Union[str, "S3DataStoreFile"],
        force_download: bool = False,
        new_if_not_exist: bool = False,
    ):
        if isinstance(file_name, S3DataStoreFile):
            file_name = file_name.file_name

        has_remote_copy = self.file_has_remote_copy(file_name)
        has_local_copy = self.file_has_local_copy(file_name)

        if force_download:
            if not has_remote_copy:
                msg = f'Could not find file "{file_name}" in remote.'
                raise KeyError(msg)
            self.download_file(file_name=file_name)

        if (not has_remote_copy) and (not has_local_copy):
            if new_if_not_exist:
                return self.new_file(file_name=file_name)
            msg = f'Could not find file "{file_name}" in local or remote.'
            raise KeyError(msg)

        if has_local_copy:
            local_etag = self.get_local_etag(file_name)
            if has_remote_copy:
                remote_etag = self.get_remote_etag(file_name)
                if local_etag != remote_etag:
                    msg = f"Local and remote ETag are different. Using local copy. ({file_name})"
                    warnings.warn(msg)
            else:
                print("Only local copy available.")
        else:
            print(f'No local copy of file "{file_name}" available.')
            print(f"Remote copy available. Trying to download...")
            self.download_file(file_name=file_name)

        return S3DataStoreFile(s3_data_store=self, file_name=file_name)

    def new_file(self, file_name: str):
        if self.file_has_local_copy(file_name):
            msg = f'File "{file_name}" already exists locally.'
            raise ValueError(msg)
        if self.file_has_remote_copy(file_name):
            msg = f'File "{file_name}" already exists in remote.'
            raise ValueError(msg)

        return S3DataStoreFile(s3_data_store=self, file_name=file_name)

    # @contextlib.contextmanager
    # def open(self, file_name: str, *args, force_download=False, **kwargs):
    #     self.get_file(file_name=file_name, force_download=force_download)
    #     local_path = self._local_path(file_name)
    #     local_path.parent.mkdir(parents=True, exist_ok=True)

    #     f = open(local_path, *args, **kwargs)
    #     try:
    #         yield f
    #     finally:
    #         f.close()

    def file_has_local_copy(self, file_name: Union[str, "S3DataStoreFile"]):
        if isinstance(file_name, S3DataStoreFile):
            file_name = file_name.file_name
        return self._local_path(file_name).exists()

    def file_has_remote_copy(self, file_name: Union[str, "S3DataStoreFile"]):
        if isinstance(file_name, S3DataStoreFile):
            file_name = file_name.file_name
        return any([f.file_name == file_name for f in self.list_remote_files()])

    def file_has_matching_etags(self, file_name: Union[str, "S3DataStoreFile"]):
        if isinstance(file_name, S3DataStoreFile):
            file_name = file_name.file_name
        has_local_copy = self.file_has_local_copy(file_name)
        has_remote_copy = self.file_has_remote_copy(file_name)

        if not has_local_copy and not has_remote_copy:
            msg = f"Neither local nor remote copy available."
            raise ValueError(msg)

        if has_local_copy and not has_remote_copy:
            msg = f"No remote copy to compare to."
            raise ValueError(msg)
        if has_remote_copy and not has_local_copy:
            msg = f"No local copy to compare to."
            raise ValueError(msg)

        return self.get_local_etag(file_name=file_name) == self.get_remote_etag(
            file_name=file_name
        )

    def get_remote_etag(self, file_name: str):
        return self.bucket.Object(key=self.key_from_file_name(file_name)).e_tag

    def get_local_etag(self, file_name: str):
        return calculate_s3_etag(file_path=self._local_path(file_name=file_name))

    def _list_local_files(self):
        return [f for f in self.local_directory.rglob("*") if f.is_file()]

    def key_from_file_name(self, file: Union[str, "S3DataStoreFile"]):
        if not isinstance(file, S3DataStoreFile):
            key = f"{self.prefix}{file}"
        else:
            key = file.key

        return key

    def _local_path(self, file_name):
        return self.local_directory / file_name
