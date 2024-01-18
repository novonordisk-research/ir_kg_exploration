"""
!!! ATTENTION !!!

Running the below tests requires a .env file in this files directory, i.e. "./.env" relative to this file.
The .env file must define the following variables:

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_SESSION_TOKEN
- BUCKET_NAME
- PREFIX

!!!>>> WARNING <<<!!!
ALL files matching BUCKET_NAME/PREFIX will be DELETED after running the tests. DO NOT use a prefix that already
exists under BUCKET_NAME; instead, use a new prefix for testing only, e.g., "mybucket/my_project/testing".
!!!>>> WARNING <<<!!!
"""

import pytest
import tempfile
import uuid
from pathlib import Path

from dotenv import dotenv_values

from anngel.datalib.s3datastore import S3DataStore


@pytest.fixture(scope="module")
def _temporary_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture()
def temporary_directory(_temporary_directory):
    tmp_dir = Path(_temporary_directory) / str(uuid.uuid4())
    tmp_dir.mkdir()
    return tmp_dir


@pytest.fixture()
def temporary_file(_temporary_directory):
    return Path(_temporary_directory) / str(uuid.uuid4())


@pytest.fixture(scope="module")
def config():
    return dotenv_values(Path(__file__).parent / ".env")


@pytest.fixture(scope="module", autouse=True)
def cleanup_s3(config):
    import boto3
    import boto3.s3

    # Cleanup before tests
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=config.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=config.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=config.get("AWS_SESSION_TOKEN"),
    )
    bucket = s3.Bucket(config["BUCKET_NAME"])
    for o in bucket.objects.filter(Prefix=config["PREFIX"]):
        o.delete()

    yield

    # Cleanup after tests
    for o in bucket.objects.filter(Prefix=config["PREFIX"]):
        o.delete()


@pytest.fixture()
def s3ds(temporary_directory, config):
    return S3DataStore(
        aws_access_key_id=config.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=config.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=config.get("AWS_SESSION_TOKEN"),
        local_directory=temporary_directory,
        bucket_name=config["BUCKET_NAME"],
        prefix=config["PREFIX"],
    )


def test_simple(temporary_directory, config):
    ds = S3DataStore(
        aws_access_key_id=config.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=config.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=config.get("AWS_SESSION_TOKEN"),
        local_directory=temporary_directory,
        bucket_name=config["BUCKET_NAME"],
        prefix=config["PREFIX"],
    )

    f_name1 = "test.txt"
    content1 = "Hello World!\n"
    ds_f1 = ds.new_file(f_name1)
    with open(ds_f1.local_path, "w") as f:
        f.write(content1)
    assert ds_f1.has_remote_copy() == False

    ds_f1.upload()
    assert ds_f1.has_remote_copy() == True
    assert ds_f1.has_matching_etags()
    ds_f1.delete(delete_remote=False)
    with pytest.raises(ValueError) as e:
        ds_f1.has_matching_etags()

    ds_f1.download()
    assert ds_f1.has_remote_copy() == True
    assert ds_f1.has_matching_etags()
    ds_f1.delete(delete_remote=False)

    ds_f2 = ds.get_file(f_name1)
    with open(ds_f2.local_path, "r") as f:
        assert f.read() == content1


import os


@pytest.mark.skipif(
    not os.environ.get("TEST_AWS_CREDENTIALS", False),
    reason="Credentials might not be available",
)
def test_auto_credentials(config, temporary_directory):
    ds = S3DataStore(
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
        local_directory=temporary_directory,
        bucket_name=config["BUCKET_NAME"],
        prefix=config["PREFIX"],
    )
    ds.list_files()
