from tqdm import tqdm
import urllib.request
from zipfile import ZipFile


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def unzip(path, output_path):
    with ZipFile(file=path) as zip_file:
        # Loop over each file
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            # Extract each file to another directory
            # If you want to extract to current working directory, don't specify path
            zip_file.extract(member=file, path=output_path)


def gunzip(path, output_path):
    import gzip
    import shutil

    print(output_path, path)

    with open(output_path, "wb") as fout:
        with gzip.open(path, "rb") as fin:
            shutil.copyfileobj(fin, fout)


def tarextract(path, output_path):
    import tarfile

    if str(path).endswith("tar.gz"):
        tar = tarfile.open(path, "r:gz")
        tar.extractall(output_path)
        tar.close()
    elif str(path).endswith("tar"):
        tar = tarfile.open(path, "r:")
        tar.extractall(output_path)
        tar.close()
