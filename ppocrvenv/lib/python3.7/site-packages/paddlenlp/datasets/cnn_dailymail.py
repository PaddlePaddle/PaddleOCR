# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import os
import hashlib
import shutil

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url, _decompress, _get_unique_endpoints
from paddle.distributed import ParallelEnv
from paddlenlp.utils.env import DATA_HOME
from paddlenlp.utils.log import logger
from . import DatasetBuilder


class CnnDailymail(DatasetBuilder):
    """
    CNN/DailyMail non-anonymized summarization dataset.
    The CNN / DailyMail Dataset is an English-language dataset containing
    just over 300k unique news articles as written by journalists at CNN
    nd the Daily Mail. The current version supports both extractive and
    abstractive summarization, though the original version was created
    for machine reading and comprehension and abstractive question answering.

    Version 1.0.0 aimed to support supervised neural methodologies for machine
    reading and question answering with a large amount of real natural language
    training data and released about 313k unique articles and nearly 1M Cloze
    style questions to go with the articles.
    Versions 2.0.0 and 3.0.0 changed the structure of the dataset to support
    summarization rather than question answering. Version 3.0.0 provided a
    non-anonymized version of the data, whereas both the previous versions were
    preprocessed to replace named entities with unique identifier labels.

    An updated version of the code that does not anonymize the data is available
    at https://github.com/abisee/cnn-dailymail.
    """
    lazy = False
    META_INFO = collections.namedtuple("META_INFO", ("file", "url", "md5"))
    SPLITS = {
        "train": META_INFO(
            "all_train.txt",
            "https://bj.bcebos.com/paddlenlp/datasets/cnn_dailymail/all_train.txt",
            "c8ca98cfcb6cf3f99a404552568490bc"),
        "dev": META_INFO(
            "all_val.txt",
            "https://bj.bcebos.com/paddlenlp/datasets/cnn_dailymail/all_val.txt",
            "83a3c483b3ed38b1392285bed668bfee"),
        "test": META_INFO(
            "all_test.txt",
            "https://bj.bcebos.com/paddlenlp/datasets/cnn_dailymail/all_test.txt",
            "4f3ac04669934dbc746b7061e68a0258")
    }
    cnn_dailymail = {
        "cnn": {
            "url":
            "https://bj.bcebos.com/paddlenlp/datasets/cnn_dailymail/cnn_stories.tgz",
            "md5": "85ac23a1926a831e8f46a6b8eaf57263",
            "file_num": 92579
        },
        "dailymail": {
            "url":
            "https://bj.bcebos.com/paddlenlp/datasets/cnn_dailymail/dailymail_stories.tgz",
            "md5": "f9c5f565e8abe86c38bfa4ae8f96fd72",
            "file_num": 219506
        }
    }

    def _read_text_file(self, text_file):
        lines = []
        with open(text_file, "r", encoding="utf8") as f:
            for line in f:
                lines.append(line.strip())
        return lines

    def _get_url_hashes(self, path):
        """Get hashes of urls in file."""
        urls = self._read_text_file(path)

        def url_hash(u):
            h = hashlib.sha1()
            try:
                u = u.encode("utf-8")
            except UnicodeDecodeError:
                logger.error("Cannot hash url: %s", u)
            h.update(u)
            return h.hexdigest()

        return {url_hash(u): True for u in urls}

    def _get_hash_from_path(self, p):
        """Extract hash from path."""
        basename = os.path.basename(p)
        return basename[0:basename.find(".story")]

    def _find_files(self, dl_paths, publisher, url_dict):
        """Find files corresponding to urls."""
        if publisher == "cnn":
            top_dir = os.path.join(dl_paths["cnn"], "stories")
        elif publisher == "dailymail":
            top_dir = os.path.join(dl_paths["dailymail"], "stories")
        else:
            logger.error("Unsupported publisher: %s", publisher)
        files = sorted(os.listdir(top_dir))

        ret_files = []
        for p in files:
            if self._get_hash_from_path(p) in url_dict:
                ret_files.append(os.path.join(top_dir, p))
        return ret_files

    def _subset_filenames(self, dl_paths, split):
        """Get filenames for a particular split."""
        # Get filenames for a split.
        urls = self._get_url_hashes(dl_paths[split])
        cnn = self._find_files(dl_paths, "cnn", urls)
        dm = self._find_files(dl_paths, "dailymail", urls)
        return cnn + dm

    def _get_art_abs(self, story_file, version):
        """Get abstract (highlights) and article from a story file path."""
        # Based on https://github.com/abisee/cnn-dailymail/blob/master/
        #     make_datafiles.py

        lines = self._read_text_file(story_file)

        # The github code lowercase the text and we removed it in 3.0.0.

        # Put periods on the ends of lines that are missing them
        # (this is a problem in the dataset because many image captions don't end in
        # periods; consequently they end up in the body of the article as run-on
        # sentences)
        def fix_missing_period(line):
            """Adds a period to a line that is missing a period."""
            if "@highlight" in line:
                return line
            if not line:
                return line
            if line[-1] in [
                    ".", "!", "?", "...", "'", "`", '"', "\u2019", "\u201d", ")"
            ]:
                return line
            return line + " ."

        lines = [fix_missing_period(line) for line in lines]

        # Separate out article and abstract sentences
        article_lines = []
        highlights = []
        next_is_highlight = False
        for line in lines:
            if not line:
                continue  # empty line
            elif line.startswith("@highlight"):
                next_is_highlight = True
            elif next_is_highlight:
                highlights.append(line)
            else:
                article_lines.append(line)

        # Make article into a single string
        article = " ".join(article_lines)

        if version >= "2.0.0":
            abstract = "\n".join(highlights)
        else:
            abstract = " ".join(highlights)

        return article, abstract

    def _get_data(self, mode):
        """ Check and download Dataset """
        dl_paths = {}
        version = self.config.get("version", "3.0.0")
        if version not in ["1.0.0", "2.0.0", "3.0.0"]:
            raise ValueError("Unsupported version: %s" % version)
        dl_paths["version"] = version
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        for k, v in self.cnn_dailymail.items():
            dir_path = os.path.join(default_root, k)
            if not os.path.exists(dir_path):
                get_path_from_url(v["url"], default_root, v["md5"])
            unique_endpoints = _get_unique_endpoints(ParallelEnv()
                                                     .trainer_endpoints[:])
            if ParallelEnv().current_endpoint in unique_endpoints:
                file_num = len(os.listdir(os.path.join(dir_path, "stories")))
                if file_num != v["file_num"]:
                    logger.warning(
                        "Number of %s stories is %d != %d, decompress again." %
                        (k, file_num, v["file_num"]))
                    shutil.rmtree(os.path.join(dir_path, "stories"))
                    _decompress(
                        os.path.join(default_root, os.path.basename(v["url"])))
            dl_paths[k] = dir_path
        filename, url, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(url, default_root, data_hash)
        dl_paths[mode] = fullname
        return dl_paths

    def _read(self, dl_paths, split):
        files = self._subset_filenames(dl_paths, split)
        for p in files:
            article, highlights = self._get_art_abs(p, dl_paths["version"])
            if not article or not highlights:
                continue
            yield {
                "article": article,
                "highlights": highlights,
                "id": self._get_hash_from_path(p),
            }
