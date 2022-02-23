# Copyright (c) 2020 VisualDL Authors. All Rights Reserve.
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
# =======================================================================

import os
import time
import numpy as np
from visualdl.writer.record_writer import RecordFileWriter
from visualdl.server.log import logger
from visualdl.utils.img_util import merge_images
from visualdl.utils.figure_util import figure_to_image
from visualdl.utils.md5_util import md5
from visualdl.component.base_component import scalar, image, embedding, audio, \
    histogram, pr_curve, roc_curve, meta_data, text, hparam


class DummyFileWriter(object):
    """A fake file writer that writes nothing to the disk.
    """

    def __init__(self, logdir):
        self._logdir = logdir

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._logdir

    def add_event(self, event, step=None, walltime=None):
        return

    def add_summary(self, summary, global_step=None, walltime=None):
        return

    def add_graph(self, graph_profile, walltime=None):
        return

    def add_onnx_graph(self, graph, walltime=None):
        return

    def flush(self):
        return

    def close(self):
        return

    def reopen(self):
        return


class LogWriter(object):
    """Log writer to write vdl records to log file.

    The class `LogWriter` provides APIs to create record file and add records to
    it. The class updates log file asynchronously without slowing down training.
    """

    def __init__(self,
                 logdir=None,
                 comment='',
                 max_queue=10,
                 flush_secs=120,
                 filename_suffix='',
                 write_to_disk=True,
                 display_name='',
                 file_name='',
                 **kwargs):
        """Create a instance of class `LogWriter` and create a vdl log file with
        given args.

        Args:
            logdir (string): Directory of log file. Default is
                `runs/**current_time**.**comment**`.
            comment (string): Suffix appended to the default `logdir`.It has no
                effect if `logidr` is assigned.
            max_queue (int): Size of queue for pending records.
            flush_secs (int): The duration to flush the pending records in queue
                to disk.
            filename_suffix (string): Suffix added to vdl log file.
            write_to_disk (boolean): Write to disk if it is True.
        """
        if not logdir:
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            if '' != comment:
                comment = '.' + comment
            logdir = os.path.join('runs', current_time + comment)
        self._logdir = logdir
        self._max_queue = max_queue
        self._flush_secs = flush_secs
        self._filename_suffix = filename_suffix
        self._write_to_disk = write_to_disk
        self.kwargs = kwargs
        self._file_name = file_name

        self._file_writer = None
        self._all_writers = {}
        self._get_file_writer()
        self.loggers = {}
        self.add_meta(display_name=display_name)

    @property
    def logdir(self):
        return self._logdir

    def _get_file_writer(self):
        if not self._write_to_disk:
            self._file_writer = DummyFileWriter(logdir=self._logdir)
            self._all_writers.update({self._logdir: self._file_writer})
            return self._file_writer

        if self._all_writers is {} or self._file_writer is None:
            self._file_writer = RecordFileWriter(
                logdir=self._logdir,
                max_queue_size=self._max_queue,
                flush_secs=self._flush_secs,
                filename_suffix=self._filename_suffix,
                filename=self._file_name)
            self._all_writers.update({self._logdir: self._file_writer})
        return self._file_writer

    @property
    def file_name(self):
        return self._file_writer.get_filename()

    def add_meta(self, tag='meta_data_tag', display_name='', step=0, walltime=None):
        """Add a meta to vdl record file.

        Args:
            tag (string): Data identifier
            display_name (string): Display name of `runs`.
            step (int): Step of meta.
            walltime (int): Wall time of scalar
        """
        if '%' in tag:
            raise RuntimeError("% can't appear in tag!")
        walltime = round(time.time() * 1000) if walltime is None else walltime
        self._get_file_writer().add_record(
            meta_data(tag=tag, display_name=display_name, step=step,
                      walltime=walltime))

    def add_scalar(self, tag, value, step, walltime=None):
        """Add a scalar to vdl record file.

        Args:
            tag (string): Data identifier
            value (float): Value of scalar
            step (int): Step of scalar
            walltime (int): Wall time of scalar

        Example:
            for index in range(1, 101):
                writer.add_scalar(tag="train/loss", value=index*0.2, step=index)
                writer.add_scalar(tag="train/lr", value=index*0.5, step=index)
        """
        if '%' in tag:
            raise RuntimeError("% can't appear in tag!")
        walltime = round(time.time() * 1000) if walltime is None else walltime
        self._get_file_writer().add_record(
            scalar(tag=tag, value=value, step=step, walltime=walltime))

    def add_image(self, tag, img, step, walltime=None, dataformats="HWC"):
        """Add an image to vdl record file.

        Args:
            tag (string): Data identifier
            img (np.ndarray): Image represented by a numpy.array
            step (int): Step of image
            walltime (int): Wall time of image
            dataformats (string): Format of image

        Example:
            from PIL import Image
            import numpy as np

            I = Image.open("./test.png")
            I_array = np.array(I)
            writer.add_image(tag="lll", img=I_array, step=0)
        """
        if '%' in tag:
            raise RuntimeError("% can't appear in tag!")
        walltime = round(time.time() * 1000) if walltime is None else walltime
        self._get_file_writer().add_record(
            image(tag=tag, image_array=img, step=step, walltime=walltime,
                  dataformats=dataformats))

    def add_figure(self, tag, figure, step, walltime=None):
        """Add an figure to vdl record file.

        Args:
            tag (string): Data identifier
            figure (matplotlib.figure.Figure): Image represented by a Figure
            step (int): Step of image
            walltime (int): Wall time of image
            dataformats (string): Format of image

        Example:
            form matplotlib import pyplot as plt
            import numpy as np

            x = np.arange(100)
            y = x ** 2 + 1
            plt.plot(x, y)
            fig = plt.gcf()
            writer.add_figure(tag="lll", figure=fig, step=0)
        """
        if '%' in tag:
            raise RuntimeError("% can't appear in tag!")
        walltime = round(time.time() * 1000) if walltime is None else walltime
        img = figure_to_image(figure)
        self._get_file_writer().add_record(
            image(tag=tag, image_array=img, step=step, walltime=walltime))

    def add_text(self, tag, text_string, step=None, walltime=None):
        """Add an text to vdl record file.
        Args:
            tag (string): Data identifier
            text_string (string): Value of text
            step (int): Step of text
            walltime (int): Wall time of text
        Example:
            for index in range(1, 101):
                writer.add_text(tag="train/loss", text_string=str(index) + 'text', step=index)
        """
        if '%' in tag:
            raise RuntimeError("% can't appear in tag!")
        walltime = round(
            time.time() * 1000) if walltime is None else walltime
        self._get_file_writer().add_record(
            text(
                tag=tag, text_string=text_string, step=step,
                walltime=walltime))

    def add_image_matrix(self, tag, imgs, step, rows=-1, scale=1.0, walltime=None, dataformats="HWC"):
        """Add an image to vdl record file.

        Args:
            tag (string): Data identifier
            imgs (np.ndarray): Image represented by a numpy.array
            step (int): Step of image
            rows (int): Number of rows, -1 means as close as possible to the square
            scale (float): Image zoom scale
            walltime (int): Wall time of image
            dataformats (string): Format of image

        Example:
            from PIL import Image
            import numpy as np

            I = Image.open("./test.png")
            I_array = np.array([I, I, I])
            writer.add_image_matrix(tag="lll", imgs=I_array, step=0)
        """
        if '%' in tag:
            raise RuntimeError("% can't appear in tag!")
        walltime = round(time.time() * 1000) if walltime is None else walltime
        img = merge_images(imgs=imgs, dataformats=dataformats, scale=scale, rows=rows)
        self.add_image(tag=tag,
                       img=img,
                       step=step,
                       walltime=walltime,
                       dataformats=dataformats)

    def add_embeddings(self, tag, mat=None, metadata=None,
                       metadata_header=None, walltime=None, labels=None,
                       hot_vectors=None, labels_meta=None):
        """Add embeddings to vdl record file.

        Args:
            tag (string): Data identifier
            mat (numpy.array or list): A matrix which each row is
                feature of labels.
            metadata (numpy.array or list): A 1D or 2D matrix of labels
            metadata_header (numpy.array or list): Meta data of labels.
            walltime (int): Wall time of embeddings.
            labels (numpy.array or list): Obsolete parameter, use `metadata` to
                replace it.
            hot_vectors (numpy.array or list): Obsolete parameter, use `mat` to
                replace it.
            labels_meta (numpy.array or list): Obsolete parameter, use
                `metadata_header` to replace it.
        Example 1:
            mat = [
            [1.3561076367500755, 1.3116267195134017, 1.6785401875616097],
            [1.1039614644440658, 1.8891609992484688, 1.32030488587171],
            [1.9924524852447711, 1.9358920727142739, 1.2124401279391606],
            [1.4129542689796446, 1.7372166387197474, 1.7317806077076527],
            [1.3913371800587777, 1.4684674577930312, 1.5214136352476377]]

            metadata = ["label_1", "label_2", "label_3", "label_4", "label_5"]
            # or like this
            # metadata = [["label_1", "label_2", "label_3", "label_4", "label_5"]]

            writer.add_embeddings(tag='default',
                                  metadata=metadata,
                                  mat=mat,
                                  walltime=round(time.time() * 1000))

        Example 2:
            mat = [
            [1.3561076367500755, 1.3116267195134017, 1.6785401875616097],
            [1.1039614644440658, 1.8891609992484688, 1.32030488587171],
            [1.9924524852447711, 1.9358920727142739, 1.2124401279391606],
            [1.4129542689796446, 1.7372166387197474, 1.7317806077076527],
            [1.3913371800587777, 1.4684674577930312, 1.5214136352476377]]

            metadata = [["label_a_1", "label_a_2", "label_a_3", "label_a_4", "label_a_5"],
                      ["label_b_1", "label_b_2", "label_b_3", "label_b_4", "label_b_5"]]

            metadata_header = ["label_a", "label_2"]

            writer.add_embeddings(tag='default',
                                  metadata=metadata,
                                  metadata_header=metadata_header,
                                  mat=mat,
                                  walltime=round(time.time() * 1000))
        """
        if '%' in tag:
            raise RuntimeError("% can't appear in tag!")
        if (mat is None) and hot_vectors:
            mat = hot_vectors
            logger.warning('Parameter `hot_vectors` in function '
                           '`add_embeddings` will be deprecated in '
                           'future, use `mat` instead.')
        if (metadata is None) and labels:
            metadata = labels
            logger.warning(
                'Parameter `labels` in function `add_embeddings` will be '
                'deprecated in future, use `metadata` instead.')
        if (metadata_header is None) and labels_meta:
            metadata_header = labels_meta
            logger.warning(
                'Parameter `labels_meta` in function `add_embeddings` will be'
                ' deprecated in future, use `metadata_header` instead.')
        if isinstance(mat, np.ndarray):
            mat = mat.tolist()
        if isinstance(metadata, np.ndarray):
            metadata = metadata.tolist()

        if isinstance(metadata[0], list) and not metadata_header:
            metadata_header = ["label_%d" % i for i in range(len(metadata))]

        step = 0
        walltime = round(time.time() * 1000) if walltime is None else walltime
        self._get_file_writer().add_record(
            embedding(
                tag=tag,
                labels=metadata,
                labels_meta=metadata_header,
                hot_vectors=mat,
                step=step,
                walltime=walltime))

    def add_audio(self,
                  tag,
                  audio_array,
                  step,
                  sample_rate=8000,
                  walltime=None):
        """Add an audio to vdl record file.

        Args:
            tag (string): Data identifier
            audio (np.ndarray or list): audio represented by a numpy.array
            step (int): Step of audio
            sample_rate (int): Sample rate of audio
            walltime (int): Wall time of audio

        Example:
            import wave

            CHUNK = 4096
            f = wave.open(audio_path, "rb")
            wavdata = []
            chunk = f.readframes(CHUNK)
            while chunk:
                data = np.frombuffer(chunk, dtype='uint8')
                wavdata.extend(data)
                chunk = f.readframes(CHUNK)
            audio_data = np.array(wavdata)

            writer.add_audio(tag="audio_test",
                             audio_array=audio_data,
                             step=0,
                             sample_rate=8000)
        """
        if '%' in tag:
            raise RuntimeError("% can't appear in tag!")
        walltime = round(time.time() * 1000) if walltime is None else walltime
        if isinstance(audio_array, list):
            audio_array = np.array(audio_array)
        self._get_file_writer().add_record(
            audio(
                tag=tag,
                audio_array=audio_array,
                sample_rate=sample_rate,
                step=step,
                walltime=walltime))

    def add_histogram(self,
                      tag,
                      values,
                      step,
                      walltime=None,
                      buckets=10):
        """Add an histogram to vdl record file.

        Args:
            tag (string): Data identifier
            value (np.ndarray or list): value represented by a numpy.array or list
            step (int): Step of histogram
            walltime (int): Wall time of audio
            buckets (int): Number of buckets, default is 10

        Example:
            values = np.arange(0, 1000)
            with LogWriter(logdir="./log/histogram_test/train") as writer:
                for index in range(5):
                    writer.add_histogram(tag='default',
                                         values=values+index,
                                         step=index)
        """
        if '%' in tag:
            raise RuntimeError("% can't appear in tag!")
        hist, bin_edges = np.histogram(values, bins=buckets)
        walltime = round(time.time() * 1000) if walltime is None else walltime
        self._get_file_writer().add_record(
            histogram(
                tag=tag,
                hist=hist,
                bin_edges=bin_edges,
                step=step,
                walltime=walltime))

    def add_hparams(self, hparams_dict, metrics_list, walltime=None):
        """Add an histogram to vdl record file.

        Args:
            hparams_dict (dictionary): Each key-value pair in the dictionary is the
              name of the hyper parameter and it's corresponding value. The type of the value
              can be one of `string`, `float` or `int`.
            metrics_list (list): Name of all metrics.
            walltime (int): Wall time of hparams.

        Examples::
            from visualdl import LogWriter

            # Remember use add_scalar to log your metrics data!
            with LogWriter('./log/hparams_test/train/run1') as writer:
                writer.add_hparams({'lr': 0.1, 'bsize': 1, 'opt': 'sgd'}, ['hparam/accuracy', 'hparam/loss'])
                for i in range(10):
                    writer.add_scalar('hparam/accuracy', i, i)
                    writer.add_scalar('hparam/loss', 2*i, i)

            with LogWriter('./log/hparams_test/train/run2') as writer:
                writer.add_hparams({'lr': 0.2, 'bsize': 2, 'opt': 'relu'}, ['hparam/accuracy', 'hparam/loss'])
                for i in range(10):
                    writer.add_scalar('hparam/accuracy', 1.0/(i+1), i)
                    writer.add_scalar('hparam/loss', 5*i, i)
        """
        if type(hparams_dict) is not dict:
            raise TypeError('hparam_dict should be dictionary!')
        if type(metrics_list) is not list:
            raise TypeError('metric_list should be list!')
        walltime = round(time.time() * 1000) if walltime is None else walltime

        self._get_file_writer().add_record(
            hparam(
                name=md5(self.file_name),
                hparam_dict=hparams_dict,
                metric_list=metrics_list,
                walltime=walltime))

    def add_pr_curve(self,
                     tag,
                     labels,
                     predictions,
                     step,
                     num_thresholds=10,
                     weights=None,
                     walltime=None):
        """Add an precision-recall curve to vdl record file.

        Args:
            tag (string): Data identifier
            labels (np.ndarray or list): Binary labels for each element.
            predictions (np.ndarray or list): The probability that an element
                be classified as true.
            step (int): Step of pr curve.
            weights (float): Multiple of data to display on the curve.
            num_thresholds (int): Number of thresholds used to draw the curve.
            walltime (int): Wall time of pr curve.

        Example:
            with LogWriter(logdir="./log/pr_curve_test/train") as writer:
                for index in range(3):
                    labels = np.random.randint(2, size=100)
                    predictions = np.random.rand(100)
                    writer.add_pr_curve(tag='default',
                                        labels=labels,
                                        predictions=predictions,
                                        step=index)
        """
        if '%' in tag:
            raise RuntimeError("% can't appear in tag!")
        walltime = round(time.time() * 1000) if walltime is None else walltime
        self._get_file_writer().add_record(
            pr_curve(
                tag=tag,
                labels=labels,
                predictions=predictions,
                step=step,
                walltime=walltime,
                num_thresholds=num_thresholds,
                weights=weights
                ))

    def add_roc_curve(self,
                      tag,
                      labels,
                      predictions,
                      step,
                      num_thresholds=10,
                      weights=None,
                      walltime=None):
        """Add an ROC curve to vdl record file.
        Args:
            tag (string): Data identifier
            labels (numpy.ndarray or list): Binary labels for each element.
            predictions (numpy.ndarray or list): The probability that an element
                be classified as true.
            step (int): Step of pr curve.
            weights (float): Multiple of data to display on the curve.
            num_thresholds (int): Number of thresholds used to draw the curve.
            walltime (int): Wall time of pr curve.
        Example:
            with LogWriter(logdir="./log/roc_curve_test/train") as writer:
                for index in range(3):
                    labels = np.random.randint(2, size=100)
                    predictions = np.random.rand(100)
                    writer.add_roc_curve(tag='default',
                                        labels=labels,
                                        predictions=predictions,
                                        step=index)
        """
        if '%' in tag:
            raise RuntimeError("% can't appear in tag!")
        walltime = round(time.time() * 1000) if walltime is None else walltime
        self._get_file_writer().add_record(
            roc_curve(
                tag=tag,
                labels=labels,
                predictions=predictions,
                step=step,
                walltime=walltime,
                num_thresholds=num_thresholds,
                weights=weights
                ))

    def flush(self):
        """Flush all data in cache to disk.
        """
        if self._all_writers is {}:
            return
        for writer in self._all_writers.values():
            writer.flush()

    def close(self):
        """Close all writers after flush data to disk.
        """
        if self._all_writers is {}:
            return
        for writer in self._all_writers.values():
            writer.flush()
            writer.close()
        self._file_writer = None
        self._all_writers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
