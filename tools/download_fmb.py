"""Download FMB directly from the authors' public Google Drive folder.

Source (SegMiF repo, checked 2026-11-01):
  https://drive.google.com/drive/folders/1T_jVi80tjgyHTQDpn-TjfySyW4CK1LlF
Students download the data themselves; we do NOT redistribute it.
If Drive blocks the download (quota / 50-file folder limit), add the folder as a
shortcut to your own Drive, mount Drive in Colab, and pass --raw to prepare_fmb.py.
"""
import argparse
import os
import tarfile
import zipfile

FOLDER_ID = "1T_jVi80tjgyHTQDpn-TjfySyW4CK1LlF"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/FMB_raw")
    a = p.parse_args()
    import gdown
    os.makedirs(a.out, exist_ok=True)
    gdown.download_folder(id=FOLDER_ID, output=a.out, quiet=False, remaining_ok=True)
    for root, _, files in os.walk(a.out):  # unpack archives if the authors zipped the data
        for f in files:
            path = os.path.join(root, f)
            if f.endswith(".zip"):
                zipfile.ZipFile(path).extractall(root)
            elif f.endswith((".tar", ".tar.gz", ".tgz")):
                tarfile.open(path).extractall(root)
    print("done ->", a.out, "(next: python tools/prepare_fmb.py --raw", a.out, ")")


if __name__ == "__main__":
    main()
