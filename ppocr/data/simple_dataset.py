# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import cv2
import math
import os
import json
import random
import traceback
import multiprocessing
import urllib.request
import urllib.parse
import threading
import concurrent.futures
from collections import OrderedDict
from paddle.io import Dataset
from .imaug import transform, create_operators
from paddle import get_device

# ------------------------------------------------------------------ #
#  Per-worker-process URL prefetch cache
#
#  Each DataLoader worker is a forked process with its own copy of
#  these globals.  The thread pool and LRU cache are therefore
#  completely independent across workers — no cross-process locking
#  needed, and memory usage is bounded per worker.
#
#  Memory budget (per worker):
#    _URL_CACHE_MAX × avg_image_size  ≈  200 × 270 KB  ≈  54 MB
#  With num_workers=4 the total extra footprint is ~216 MB.
#
#  How prefetch works:
#    _ensure_index_map() fires inside the worker when an epoch changes.
#    It calls _prefetch_epoch_urls(), which scans the new _index_map,
#    picks the first _URL_PREFETCH_SUBMIT URL entries (epoch order ≈
#    access order), and submits them to the background thread pool.
#    _load_image_bytes() checks the cache / in-flight future before
#    falling back to a synchronous download.
# ------------------------------------------------------------------ #

_URL_CACHE_MAX = 200  # max cached images per worker
_URL_PREFETCH_SUBMIT = 200  # URL items submitted to thread pool per epoch

# LRU cache: url -> bytes
_url_cache: "OrderedDict[str, bytes]" = OrderedDict()
_url_cache_lock = threading.Lock()

# In-flight futures: url -> Future
_url_futures: "dict[str, concurrent.futures.Future]" = {}
_url_futures_lock = threading.Lock()

# Lazily created per-process thread pool
_url_executor: "concurrent.futures.ThreadPoolExecutor | None" = None
_url_executor_lock = threading.Lock()


def _get_url_executor():
    global _url_executor
    if _url_executor is None:
        with _url_executor_lock:
            if _url_executor is None:
                _url_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=4,
                    thread_name_prefix="url_prefetch",
                )
    return _url_executor


def _encode_url(url):
    """Percent-encode non-ASCII characters in the URL path so that
    urllib can handle URLs containing CJK or other non-ASCII filenames.
    Scheme, netloc, query and fragment are left untouched.
    """
    parts = urllib.parse.urlparse(url)
    encoded_path = urllib.parse.quote(parts.path, safe="/:@!$&'()*+,;=")
    return urllib.parse.urlunparse(parts._replace(path=encoded_path))


def _download_url_bytes(url):
    """Download *url*, store in LRU cache, remove from futures dict.
    The futures entry is always cleaned up (success or failure) so that
    a failed URL can be retried on the next access.
    """
    encoded = _encode_url(url)
    try:
        with urllib.request.urlopen(encoded, timeout=30) as resp:
            data = resp.read()
    except Exception:
        with _url_futures_lock:
            _url_futures.pop(url, None)
        raise
    with _url_cache_lock:
        if url not in _url_cache:
            if len(_url_cache) >= _URL_CACHE_MAX:
                _url_cache.popitem(last=False)  # evict LRU entry
            _url_cache[url] = data
        else:
            _url_cache.move_to_end(url)
    with _url_futures_lock:
        _url_futures.pop(url, None)
    return data


def _submit_url_prefetch(url):
    """Submit a background download for *url* if not already cached/in-flight."""
    with _url_cache_lock:
        if url in _url_cache:
            return
    with _url_futures_lock:
        if url in _url_futures:
            return
        future = _get_url_executor().submit(_download_url_bytes, url)
        _url_futures[url] = future


def _prefetch_epoch_urls(index_map, all_lines, delimiter):
    """Scan *index_map* and submit the first _URL_PREFETCH_SUBMIT URL
    items for background download.  Called inside worker processes."""
    submitted = 0
    for file_idx in index_map:
        if submitted >= _URL_PREFETCH_SUBMIT:
            break
        try:
            line = all_lines[file_idx].decode("utf-8")
            fname = line.strip("\n").split(delimiter)[0]
            if fname and fname[0] == "[":  # JSON list — skip
                continue
            if fname.startswith("http://") or fname.startswith("https://"):
                _submit_url_prefetch(fname)
                submitted += 1
        except Exception:
            pass


def _load_image_bytes(img_path):
    """Return raw image bytes.  For URLs checks prefetch cache/future first."""
    if img_path.startswith("http://") or img_path.startswith("https://"):
        # 1. Cache hit — return immediately
        with _url_cache_lock:
            if img_path in _url_cache:
                _url_cache.move_to_end(img_path)
                return _url_cache[img_path]
        # 2. In-flight future — wait for background download to finish
        with _url_futures_lock:
            future = _url_futures.get(img_path)
        if future is not None:
            return future.result(timeout=60)
        # 3. Cold miss — download synchronously (also fills cache)
        return _download_url_bytes(img_path)
    with open(img_path, "rb") as f:
        return f.read()


def _img_path_exists(img_path):
    """Return True if the image source is accessible (local file exists or URL)."""
    if img_path.startswith("http://") or img_path.startswith("https://"):
        return True
    return os.path.exists(img_path)


class SimpleDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]

        self.delimiter = dataset_config.get("delimiter", "\t")
        label_file_list = dataset_config.pop("label_file_list")
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        self.label_file_list = label_file_list
        self.ratio_list = ratio_list

        assert (
            len(ratio_list) == data_source_num
        ), "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config["data_dir"]
        self.do_shuffle = loader_config["shuffle"]
        self.seed = seed
        self.need_reset = True in [x < 1 for x in ratio_list]

        logger.info("Initialize indexs of datasets:%s" % label_file_list)

        if self.need_reset:
            # Pre-load all lines once (immutable, never re-read from disk).
            # Per-epoch ratio sampling is done via _index_map (virtual idx -> global idx).
            self._all_lines, self.file_boundaries = self._load_all_lines(
                label_file_list
            )
            self._index_map = self._generate_index_map(seed)
            self._cached_epoch = seed if seed is not None else 0
            # data_lines / data_idx_order_list kept for API compat but NOT used in __getitem__
            self.data_lines = self._all_lines
            self.data_idx_order_list = list(range(len(self._index_map)))
        else:
            self._all_lines = None
            self._index_map = None
            self._cached_epoch = None
            self.file_boundaries = None
            self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
            self.data_idx_order_list = list(range(len(self.data_lines)))
            if self.mode == "train" and self.do_shuffle:
                self.shuffle_data_random()

        # Shared epoch value: workers read this via shared memory to detect epoch changes
        self._shared_epoch = multiprocessing.Value("i", seed if seed is not None else 0)

        self.ops = create_operators(dataset_config["transforms"], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 2)

    # ------------------------------------------------------------------ #
    #  Data loading helpers
    # ------------------------------------------------------------------ #

    def _load_all_lines(self, file_list):
        """Read all label files once. Returns (all_lines, file_boundaries)."""
        if isinstance(file_list, str):
            file_list = [file_list]
        all_lines = []
        boundaries = [0]
        for f in file_list:
            with open(f, "rb") as fh:
                lines = fh.readlines()
                all_lines.extend(lines)
                boundaries.append(len(all_lines))
        return all_lines, boundaries

    def _generate_index_map(self, seed):
        """Generate virtual-index -> global-index mapping.

        Replicates the EXACT sampling logic of original get_image_info_list +
        shuffle_data_random: for each file, random.seed(seed) then
        random.sample to pick indices, then random.seed(seed) + shuffle.

        Since random.sample(population, k) with the same seed selects the
        same POSITIONS regardless of population type, sampling from
        range(start, end) yields the same positions as from lines[start:end].
        """
        sampled = []
        for i in range(len(self.ratio_list)):
            start = self.file_boundaries[i]
            end = self.file_boundaries[i + 1]
            file_size = end - start
            count = round(file_size * self.ratio_list[i])
            if self.mode == "train" or self.ratio_list[i] < 1.0:
                random.seed(seed)
                sampled.extend(random.sample(range(start, end), count))
            else:
                sampled.extend(range(start, end))
        if self.mode == "train" and self.do_shuffle:
            random.seed(seed)
            random.shuffle(sampled)
        return sampled

    def _ensure_index_map(self):
        """Lazily rebuild _index_map when worker detects epoch change via shared memory.
        Also triggers URL prefetch on first call (epoch 0) and on every epoch change.
        """
        if self._all_lines is None:
            return
        current_epoch = self._shared_epoch.value
        epoch_changed = current_epoch != self._cached_epoch
        first_call = not getattr(self, "_url_prefetch_initialized", False)

        if epoch_changed:
            self._index_map = self._generate_index_map(current_epoch)
            self._cached_epoch = current_epoch

        if epoch_changed or first_call:
            self._url_prefetch_initialized = True
            _prefetch_epoch_urls(self._index_map, self._all_lines, self.delimiter)

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines, round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    # ------------------------------------------------------------------ #
    #  Epoch update (called from main process each epoch)
    # ------------------------------------------------------------------ #

    def reset_data_lines(self, seed=None, epoch=None):
        """Signal new epoch to persistent workers via shared memory.

        Workers lazily rebuild their _index_map on next __getitem__ call.
        No disk I/O, no dataloader reconstruction.
        """
        self.seed = seed
        epoch_val = epoch if epoch is not None else (seed if seed is not None else 0)
        self._shared_epoch.value = int(epoch_val)

        if self._all_lines is not None:
            # Update main-process index_map (used by len() and batch_sampler)
            self._index_map = self._generate_index_map(seed)
            self._cached_epoch = int(epoch_val)
            self.data_idx_order_list = list(range(len(self._index_map)))
        else:
            # Fallback for non-ratio cases
            self.data_lines = self.get_image_info_list(
                self.label_file_list, self.ratio_list
            )
            self.data_idx_order_list = list(range(len(self.data_lines)))
            if self.mode == "train" and self.do_shuffle:
                self.shuffle_data_random()

    # ------------------------------------------------------------------ #
    #  Data access
    # ------------------------------------------------------------------ #

    def _try_parse_filename_list(self, file_name):
        # multiple images -> one gt label
        if len(file_name) > 0 and file_name[0] == "[":
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except:
                pass
        return file_name

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, "ext_data_num"):
                ext_data_num = getattr(op, "ext_data_num")
                break
        load_data_ops = self.ops[: self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            if self._index_map is not None:
                # Sample from current epoch's subset (same as original)
                self._ensure_index_map()
                rand_virtual = np.random.randint(len(self._index_map))
                file_idx = self._index_map[rand_virtual]
                data_line = self._all_lines[file_idx]
            else:
                file_idx = self.data_idx_order_list[np.random.randint(self.__len__())]
                data_line = self.data_lines[file_idx]
            data_line = data_line.decode("utf-8")
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = (
                file_name
                if file_name.startswith("http://") or file_name.startswith("https://")
                else os.path.join(self.data_dir, file_name)
            )
            data = {"img_path": img_path, "label": label}
            if not _img_path_exists(img_path):
                continue
            try:
                data["image"] = _load_image_bytes(img_path)
            except Exception:
                continue
            data = transform(data, load_data_ops)

            if data is None:
                continue
            if "polys" in data.keys():
                if data["polys"].shape[1] != 4:
                    continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        if self._index_map is not None:
            self._ensure_index_map()
            file_idx = self._index_map[idx]
            data_line = self._all_lines[file_idx]
        else:
            file_idx = self.data_idx_order_list[idx]
            data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode("utf-8")
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = (
                file_name
                if file_name.startswith("http://") or file_name.startswith("https://")
                else os.path.join(self.data_dir, file_name)
            )
            data = {"img_path": img_path, "label": label}
            if not _img_path_exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            data["image"] = _load_image_bytes(img_path)
            data["ext_data"] = self.get_ext_data()
            data["filename"] = data["img_path"]
            data["epoch"] = self._shared_epoch.value
            outs = transform(data, self.ops)
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()
                )
            )
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = (
                np.random.randint(self.__len__())
                if self.mode == "train"
                else (idx + 1) % self.__len__()
            )
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        if self._index_map is not None:
            return len(self._index_map)
        return len(self.data_idx_order_list)


class MultiScaleDataSet(SimpleDataSet):
    def __init__(self, config, mode, logger, seed=None):
        super(MultiScaleDataSet, self).__init__(config, mode, logger, seed)
        self.ds_width = config[mode]["dataset"].get("ds_width", False)
        if self.ds_width:
            self.wh_aware()

    def wh_aware(self):
        data_line_new = []
        wh_ratio = []
        for line in self.data_lines:
            data_line_new.append(line)
            line = line.decode("utf-8")
            name, label, w, h = line.strip("\n").split(self.delimiter)
            wh_ratio.append(float(w) / float(h))

        self.data_lines = data_line_new
        self.wh_ratio = np.array(wh_ratio)
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.data_idx_order_list = list(range(len(self.data_lines)))

    def resize_norm_img(self, data, imgW, imgH, padding=True):
        img = data["image"]
        h = img.shape[0]
        w = img.shape[1]
        if not padding:
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=cv2.INTER_LINEAR
            )
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")

        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        valid_ratio = min(1.0, float(resized_w / imgW))
        data["image"] = padding_im
        data["valid_ratio"] = valid_ratio
        if "iluvatar_gpu" in get_device():
            data["valid_ratio"] = np.float32(valid_ratio)
        return data

    def __getitem__(self, properties):
        # properties is a tuple, contains (width, height, index)
        img_height = properties[1]
        idx = properties[2]
        if self.ds_width and properties[3] is not None:
            wh_ratio = properties[3]
            img_width = img_height * (
                1 if int(round(wh_ratio)) == 0 else int(round(wh_ratio))
            )
            file_idx = self.wh_ratio_sort[idx]
        else:
            file_idx = self.data_idx_order_list[idx]
            img_width = properties[0]
            wh_ratio = None

        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode("utf-8")
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = (
                file_name
                if file_name.startswith("http://") or file_name.startswith("https://")
                else os.path.join(self.data_dir, file_name)
            )
            data = {"img_path": img_path, "label": label}
            if not _img_path_exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            data["image"] = _load_image_bytes(img_path)
            data["ext_data"] = self.get_ext_data()
            outs = transform(data, self.ops[:-1])
            if outs is not None:
                outs = self.resize_norm_img(outs, img_width, img_height)
                outs = transform(outs, self.ops[-1:])
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()
                )
            )
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = (idx + 1) % self.__len__()
            return self.__getitem__([img_width, img_height, rnd_idx, wh_ratio])
        return outs
