import sys
import os
import zipfile
import tarfile
import shutil
from urllib.request import urlretrieve
from anafora import copy_text
import anafora
import ssl
from bz2 import BZ2File
ssl._create_default_https_context = ssl._create_unverified_context


def prepare_practice(target_path):

    time_path = os.path.join(target_path, "time")
    # if os.path.exists(time_path):
    #     raise Exception("%s already exists!" % time_path)
    download_path = os.path.join(target_path, "TempEval-3")  # os.path.join(target_path, "download")
    os.makedirs(download_path, exist_ok=True)

    with open("practice_time_documents.txt") as list_file:
        practice_document_list = list_file.read().strip().split("\n")
        practice_document_list = list(map(os.path.normpath, practice_document_list))

    # print("Downloading tbaq-2013-03.zip...", end=" ", flush=True)
    zip_path = os.path.join(download_path, "te3.tar.bz2")
    # urlretrieve("https://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/tbaq-2013-03.zip", zip_path)
    # urlretrieve("https://figshare.com/articles/dataset/TempEval-3_data/9586532", zip_path)

    # with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #     zip_ref.extractall(download_path)
    # tbaq_path = os.path.join(download_path, "TBAQ-cleaned")
    # print("Done.")

    # print("Downloading te3-platinumstandard.tar.gz...", end=" ", flush=True)
    # tar_path = os.path.join(download_path, "te3-platinumstandard.tar.gz")
    # urlretrieve("https://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/te3-platinumstandard.tar.gz", tar_path)
    # tar = tarfile.open(tar_path)
    # tar.extractall(tbaq_path)
    # tar.close()
    # print("Done.")

    # print("Copying text...", end=" ", flush=True)
    # timeml_path = f'practice_text/TBAQ-cleaned/TempEval-tools/merge-relations/timegraph_input/wsj_0555.tml'
    # a = ''.join(anafora.ElementTree.parse(timeml_path).getroot().itertext())
    os.makedirs(time_path, exist_ok=True)
    tbaq_path = download_path # os.path.join(download_path, "TE3-platinum-test-and-data")
    copy_text.copy_timeml_text(tbaq_path, time_path, "", False)
    print("Done.")

    print("Cleaning directory...", end=" ", flush=True)
    # for root, _, files in os.walk(time_path):
    #     if files:
    #         document_rel_path = os.path.relpath(root, time_path)
    #         if document_rel_path not in practice_document_list:
    #             shutil.rmtree(root)
    # shutil.rmtree(download_path)
    print("Done.")


if __name__ == "__main__":
    path = f'practice_text/'
    prepare_practice(path)
