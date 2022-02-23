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

from visualdl.proto.record_pb2 import Record
import numpy as np
from PIL import Image


def scalar(tag, value, step, walltime=None):
    """Package data to one scalar.

    Args:
        tag (string): Data identifier
        value (float): Value of scalar
        step (int): Step of scalar
        walltime (int): Wall time of scalar

    Return:
        Package with format of record_pb2.Record
    """
    value = float(value)
    return Record(values=[
        Record.Value(id=step, tag=tag, timestamp=walltime, value=value)
    ])


def meta_data(tag='meta_data_tag', display_name="", step=0, walltime=None):
    """Package data to one meta_data.

    Meta data is info for one record file, include `display_name` etc.

    Args:
        tag (string): Data identifier
        display_name (string): Replace
        step (int): Step of scalar
        walltime (int): Wall time of scalar

    Return:
        Package with format of record_pb2.Record
    """
    meta = Record.MetaData(display_name=display_name)
    return Record(values=[
        Record.Value(id=step, tag=tag, timestamp=walltime,
                     meta_data=meta)
    ])


def imgarray2bytes(np_array):
    """Convert image ndarray to bytes.

    Args:
        np_array (np.ndarray): Array to converte.

    Returns:
        Binary bytes of np_array.
    """
    try:
        import cv2

        np_array = cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB)
        ret, buf = cv2.imencode(".png", np_array)
        img_bin = Image.fromarray(np.uint8(buf)).tobytes("raw")
    except ImportError:
        import io
        im = Image.fromarray(np_array)
        with io.BytesIO() as fp:
            im.save(fp, format='png')
            img_bin = fp.getvalue()
    return img_bin


def make_grid(I, ncols=8):  # noqa: E741
    assert isinstance(
        I, np.ndarray), 'plugin error, should pass numpy array here'
    if I.shape[1] == 1:
        I = np.concatenate([I, I, I], 1)  # noqa: E741
    assert I.ndim == 4 and I.shape[1] == 3 or I.shape[1] == 4
    nimg = I.shape[0]
    H = I.shape[2]
    W = I.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg) / ncols))
    canvas = np.zeros((I.shape[1], H * nrows, W * ncols), dtype=I.dtype)
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= nimg:
                break
            canvas[:, y * H:(y + 1) * H, x * W:(x + 1) * W] = I[i]
            i = i + 1
    return canvas


def convert_to_HWC(tensor, input_format):
    """Convert `NCHW`, `HWC`, `HW` to `HWC`

    Args:
        tensor (np.ndarray): Value of image
        input_format (string): Format of image

    Return:
        Image of format `HWC`.
    """
    assert(len(set(input_format)) == len(input_format)), "You can not use the same dimension shordhand twice. \
        input_format: {}".format(input_format)
    assert(len(tensor.shape) == len(input_format)), "size of input tensor and input format are different. \
        tensor shape: {}, input_format: {}".format(tensor.shape, input_format)
    input_format = input_format.upper()

    if len(input_format) == 4:
        index = [input_format.find(c) for c in 'NCHW']
        tensor_NCHW = tensor.transpose(index)
        tensor_CHW = make_grid(tensor_NCHW)
        return tensor_CHW.transpose(1, 2, 0)

    if len(input_format) == 3:
        index = [input_format.find(c) for c in 'HWC']
        tensor_HWC = tensor.transpose(index)
        if tensor_HWC.shape[2] == 1:
            tensor_HWC = np.concatenate([tensor_HWC, tensor_HWC, tensor_HWC], 2)
        return tensor_HWC

    if len(input_format) == 2:
        index = [input_format.find(c) for c in 'HW']
        tensor = tensor.transpose(index)
        tensor = np.stack([tensor, tensor, tensor], 2)
        return tensor


def denormalization(image_array):
    """Renormalise ndarray matrix.

    Args:
        image_array(np.ndarray): Value of image

    Return:
        Matrix after renormalising.
    """
    if image_array.max() <= 1 and image_array.min() >= 0:
        image_array *= 255
    return image_array.astype(np.uint8)


def image(tag, image_array, step, walltime=None, dataformats="HWC"):
    """Package data to one image.

    Args:
        tag (string): Data identifier
        image_array (np.ndarray): Value of image
        step (int): Step of image
        walltime (int): Wall time of image
        dataformats (string): Format of image

    Return:
        Package with format of record_pb2.Record
    """
    image_array = denormalization(image_array)
    image_array = convert_to_HWC(image_array, dataformats)
    image_bytes = imgarray2bytes(image_array)
    image = Record.Image(encoded_image_string=image_bytes)
    return Record(values=[
        Record.Value(id=step, tag=tag, timestamp=walltime, image=image)
    ])


def embedding(tag, labels, hot_vectors, step, labels_meta=None, walltime=None):
    """Package data to one embedding.

    Args:
        tag (string): Data identifier
        labels (list): A list of labels.
        hot_vectors (np.array or list): A matrix which each row is
            feature of labels.
        step (int): Step of embeddings.
        walltime (int): Wall time of embeddings.

    Return:
        Package with format of record_pb2.Record
    """
    embeddings = Record.Embeddings()

    if labels_meta:
        embeddings.label_meta.extend(labels_meta)

    if isinstance(labels[0], list):
        temp = []
        for index in range(len(labels[0])):
            temp.append([label[index] for label in labels])
        labels = temp
    for label, hot_vector in zip(labels, hot_vectors):
        if not isinstance(label, list):
            label = [label]
        embeddings.embeddings.append(Record.Embedding(label=label, vectors=hot_vector))

    return Record(values=[
        Record.Value(
            id=step, tag=tag, timestamp=walltime, embeddings=embeddings)
    ])


def audio(tag, audio_array, sample_rate, step, walltime):
    """Package data to one audio.

    Args:
        tag (string): Data identifier
        audio_array (np.ndarray or list): audio represented by a np.array
        sample_rate (int): Sample rate of audio
        step (int): Step of audio
        walltime (int): Wall time of audio

    Return:
        Package with format of record_pb2.Record
    """
    audio_array = audio_array.squeeze()
    if abs(audio_array).max() > 1:
        print('warning: audio amplitude out of range, auto clipped.')
        audio_array = audio_array.clip(-1, 1)
    assert (audio_array.ndim == 1), 'input tensor should be 1 dimensional.'

    audio_array = [int(32767.0 * x) for x in audio_array]

    import io
    import wave
    import struct

    fio = io.BytesIO()
    wave_writer = wave.open(fio, 'wb')
    wave_writer.setnchannels(1)
    wave_writer.setsampwidth(2)
    wave_writer.setframerate(sample_rate)
    audio_enc = b''
    audio_enc += struct.pack("<" + "h" * len(audio_array), *audio_array)
    wave_writer.writeframes(audio_enc)
    wave_writer.close()
    audio_string = fio.getvalue()
    fio.close()
    audio_data = Record.Audio(
        sample_rate=sample_rate,
        num_channels=1,
        length_frames=len(audio_array),
        encoded_audio_string=audio_string,
        content_type='audio/wav')
    return Record(values=[
        Record.Value(id=step, tag=tag, timestamp=walltime, audio=audio_data)
    ])


def text(tag, text_string, step, walltime=None):
    """Package data to one image.
    Args:
        tag (string): Data identifier
        text_string (string): Value of text
        step (int): Step of text
        walltime (int): Wall time of text
    Return:
        Package with format of record_pb2.Record
    """
    _text = Record.Text(encoded_text_string=text_string)
    return Record(values=[
        Record.Value(id=step, tag=tag, timestamp=walltime, text=_text)
    ])


def histogram(tag, hist, bin_edges, step, walltime):
    """Package data to one histogram.

    Args:
        tag (string): Data identifier
        hist (np.ndarray or list): The values of the histogram
        bin_edges (np.ndarray or list): The bin edges
        step (int): Step of histogram
        walltime (int): Wall time of histogram

    Return:
        Package with format of record_pb2.Record
    """
    histogram = Record.Histogram(hist=hist, bin_edges=bin_edges)
    return Record(values=[
        Record.Value(
            id=step, tag=tag, timestamp=walltime, histogram=histogram)
    ])


def hparam(name, hparam_dict, metric_list, walltime):
    """Package data to one histogram.

    Args:
        name (str): Name of hparam.
        hparam_dict (dictionary): Each key-value pair in the dictionary is the
              name of the hyper parameter and it's corresponding value. The type of the value
              can be one of `bool`, `string`, `float`, `int`, or `None`.
        metric_list (list): Name of all metrics.
        walltime (int): Wall time of hparam.

    Return:
        Package with format of record_pb2.Record
    """

    hm = Record.HParam()
    hm.name = name
    for k, v in hparam_dict.items():
        if v is None:
            continue
        hparamInfo = Record.HParam.HparamInfo()
        hparamInfo.name = k
        if isinstance(v, int):
            hparamInfo.int_value = v
            hm.hparamInfos.append(hparamInfo)
        elif isinstance(v, float):
            hparamInfo.float_value = v
            hm.hparamInfos.append(hparamInfo)
        elif isinstance(v, str):
            hparamInfo.string_value = v
            hm.hparamInfos.append(hparamInfo)
        else:
            print("The value of %s must be int, float or str, not %s" % (k, str(type(v))))
    for metric in metric_list:
        metricInfo = Record.HParam.HparamInfo()
        metricInfo.name = metric
        metricInfo.float_value = 0
        hm.metricInfos.append(metricInfo)

    return Record(values=[
        Record.Value(
            id=1, tag="hparam", timestamp=walltime, hparam=hm)
    ])


def compute_curve(labels, predictions, num_thresholds=None, weights=None):
    """ Compute precision-recall curve data by labels and predictions.

    Args:
        labels (np.ndarray or list): Binary labels for each element.
        predictions (np.ndarray or list): The probability that an element be
            classified as true.
        num_thresholds (int): Number of thresholds used to draw the curve.
        weights (float): Multiple of data to display on the curve.
    """
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    _MINIMUM_COUNT = 1e-7

    if weights is None:
        weights = 1.0

    bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
    float_labels = labels.astype(np.float)
    histogram_range = (0, num_thresholds - 1)
    tp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=float_labels * weights)
    fp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=(1.0 - float_labels) * weights)

    # Obtain the reverse cumulative sum.
    tp = np.cumsum(tp_buckets[::-1])[::-1]
    fp = np.cumsum(fp_buckets[::-1])[::-1]
    tn = fp[0] - fp
    fn = tp[0] - tp
    precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)
    recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)
    data = {
        'tp': tp.astype(int).tolist(),
        'fp': fp.astype(int).tolist(),
        'tn': tn.astype(int).tolist(),
        'fn': fn.astype(int).tolist(),
        'precision': precision.astype(float).tolist(),
        'recall': recall.astype(float).tolist()
    }
    return data


def pr_curve(tag, labels, predictions, step, walltime, num_thresholds=127,
             weights=None):
    """Package data to one pr_curve.

    Args:
        tag (string): Data identifier
        labels (np.ndarray or list): Binary labels for each element.
        predictions (np.ndarray or list): The probability that an element be
            classified as true.
        step (int): Step of pr_curve
        walltime (int): Wall time of pr_curve
        num_thresholds (int): Number of thresholds used to draw the curve.
        weights (float): Multiple of data to display on the curve.

    Return:
        Package with format of record_pb2.Record
    """
    num_thresholds = min(num_thresholds, 127)
    prcurve_map = compute_curve(labels, predictions, num_thresholds, weights)

    return pr_curve_raw(tag=tag,
                        tp=prcurve_map['tp'],
                        fp=prcurve_map['fp'],
                        tn=prcurve_map['tn'],
                        fn=prcurve_map['fn'],
                        precision=prcurve_map['precision'],
                        recall=prcurve_map['recall'],
                        step=step,
                        walltime=walltime)


def pr_curve_raw(tag, tp, fp, tn, fn, precision, recall, step, walltime):
    """Package raw data to one pr_curve.

    Args:
        tag (string): Data identifier
        tp (list): True Positive.
        fp (list): False Positive.
        tn (list): True Negative.
        fn (list): False Negative.
        precision (list): The fraction of retrieved documents that are relevant
            to the query:
        recall (list): The fraction of the relevant documents that are
            successfully retrieved.
        step (int): Step of pr_curve
        walltime (int): Wall time of pr_curve
        num_thresholds (int): Number of thresholds used to draw the curve.
        weights (float): Multiple of data to display on the curve.

    Return:
        Package with format of record_pb2.Record
    """

    """
    if isinstance(tp, np.ndarray):
        tp = tp.astype(int).tolist()
    if isinstance(fp, np.ndarray):
        fp = fp.astype(int).tolist()
    if isinstance(tn, np.ndarray):
        tn = tn.astype(int).tolist()
    if isinstance(fn, np.ndarray):
        fn = fn.astype(int).tolist()
    if isinstance(precision, np.ndarray):
        precision = precision.astype(int).tolist()
    if isinstance(recall, np.ndarray):
        recall = recall.astype(int).tolist()
    """
    prcurve = Record.PRCurve(TP=tp,
                             FP=fp,
                             TN=tn,
                             FN=fn,
                             precision=precision,
                             recall=recall)
    return Record(values=[
        Record.Value(
            id=step, tag=tag, timestamp=walltime, pr_curve=prcurve)
    ])


def compute_roc_curve(labels, predictions, num_thresholds=None, weights=None):
    """ Compute ROC curve data by labels and predictions.
    Args:
        labels (numpy.ndarray or list): Binary labels for each element.
        predictions (numpy.ndarray or list): The probability that an element be
            classified as true.
        num_thresholds (int): Number of thresholds used to draw the curve.
        weights (float): Multiple of data to display on the curve.
    """
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    _MINIMUM_COUNT = 1e-7

    if weights is None:
        weights = 1.0

    bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
    float_labels = labels.astype(np.float)
    histogram_range = (0, num_thresholds - 1)
    tp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=float_labels * weights)
    fp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=(1.0 - float_labels) * weights)

    # Obtain the reverse cumulative sum.
    tp = np.cumsum(tp_buckets[::-1])[::-1]
    fp = np.cumsum(fp_buckets[::-1])[::-1]
    tn = fp[0] - fp
    fn = tp[0] - tp
    tpr = tp / np.maximum(_MINIMUM_COUNT, tn + fp)
    fpr = fp / np.maximum(_MINIMUM_COUNT, tn + fp)
    data = {
        'tp': tp.astype(int).tolist(),
        'fp': fp.astype(int).tolist(),
        'tn': tn.astype(int).tolist(),
        'fn': fn.astype(int).tolist(),
        'tpr': tpr.astype(float).tolist(),
        'fpr': fpr.astype(float).tolist()
    }
    return data


def roc_curve(tag, labels, predictions, step, walltime, num_thresholds=127, weights=None):
    """Package data to one roc_curve.
    Args:
        tag (string): Data identifier
        labels (numpy.ndarray or list): Binary labels for each element.
        predictions (numpy.ndarray or list): The probability that an element be
            classified as true.
        step (int): Step of pr_curve
        walltime (int): Wall time of pr_curve
        num_thresholds (int): Number of thresholds used to draw the curve.
        weights (float): Multiple of data to display on the curve.
    Return:
        Package with format of record_pb2.Record
    """
    num_thresholds = min(num_thresholds, 127)
    roc_curve_map = compute_roc_curve(labels, predictions, num_thresholds, weights)

    return roc_curve_raw(tag=tag,
                         tp=roc_curve_map['tp'],
                         fp=roc_curve_map['fp'],
                         tn=roc_curve_map['tn'],
                         fn=roc_curve_map['fn'],
                         tpr=roc_curve_map['tpr'],
                         fpr=roc_curve_map['fpr'],
                         step=step,
                         walltime=walltime)


def roc_curve_raw(tag, tp, fp, tn, fn, tpr, fpr, step, walltime):
    """Package raw data to one roc_curve.
    Args:
        tag (string): Data identifier
        tp (list): True Positive.
        fp (list): False Positive.
        tn (list): True Negative.
        fn (list): False Negative.
        tpr (list): true positive rate:
        fpr (list): false positive rate.
        step (int): Step of roc_curve
        walltime (int): Wall time of roc_curve
        num_thresholds (int): Number of thresholds used to draw the curve.
        weights (float): Multiple of data to display on the curve.
    Return:
        Package with format of record_pb2.Record
    """

    """
    if isinstance(tp, np.ndarray):
        tp = tp.astype(int).tolist()
    if isinstance(fp, np.ndarray):
        fp = fp.astype(int).tolist()
    if isinstance(tn, np.ndarray):
        tn = tn.astype(int).tolist()
    if isinstance(fn, np.ndarray):
        fn = fn.astype(int).tolist()
    if isinstance(tpr, np.ndarray):
        tpr = tpr.astype(int).tolist()
    if isinstance(fpr, np.ndarray):
        fpr = fpr.astype(int).tolist()
    """
    roc_curve = Record.ROC_Curve(TP=tp,
                                 FP=fp,
                                 TN=tn,
                                 FN=fn,
                                 tpr=tpr,
                                 fpr=fpr)
    return Record(values=[
        Record.Value(
            id=step, tag=tag, timestamp=walltime, roc_curve=roc_curve)
    ])
