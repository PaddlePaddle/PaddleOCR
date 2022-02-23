from abc import ABC, abstractmethod
from enum import IntEnum
import importlib
import io
import os
import json
import csv
import warnings
import pickle

import numpy as np
import pandas as pd
from cv2 import imencode
from paddleocr import PaddleOCR

from .elements import *
from .io import load_dataframe

__all__ = ["GCVFeatureType", "GCVAgent", "TesseractFeatureType", "TesseractAgent", "PaddleocrAgent"]


def _cvt_GCV_vertices_to_points(vertices):
    return np.array([[vertex.x, vertex.y] for vertex in vertices])


class BaseOCRElementType(IntEnum):
    @property
    @abstractmethod
    def attr_name(self):
        pass


class BaseOCRAgent(ABC):
    @property
    @abstractmethod
    def DEPENDENCIES(self):
        """DEPENDENCIES lists all necessary dependencies for the class."""
        pass

    @property
    @abstractmethod
    def MODULES(self):
        """MODULES instructs how to import these necessary libraries.

        Note:
            Sometimes a python module have different installation name and module name (e.g.,
            `pip install tensorflow-gpu` when installing and `import tensorflow` when using
            ). And sometimes we only need to import a submodule but not whole module. MODULES
            is designed for this purpose.

        Returns:
            :obj: list(dict): A list of dict indicate how the model is imported.

                Example::

                    [{
                        "import_name": "_vision",
                        "module_path": "google.cloud.vision"
                    }]

                    is equivalent to self._vision = importlib.import_module("google.cloud.vision")
        """
        pass

    @classmethod
    def _import_module(cls):
        for m in cls.MODULES:
            if importlib.util.find_spec(m["module_path"]):
                setattr(
                    cls, m["import_name"], importlib.import_module(m["module_path"])
                )
            else:
                raise ModuleNotFoundError(
                    f"\n "
                    f"\nPlease install the following libraries to support the class {cls.__name__}:"
                    f"\n    pip install {' '.join(cls.DEPENDENCIES)}"
                    f"\n "
                )

    def __new__(cls, *args, **kwargs):

        cls._import_module()
        return super().__new__(cls)

    @abstractmethod
    def detect(self, image):
        pass


class GCVFeatureType(BaseOCRElementType):
    """
    The element types from Google Cloud Vision API
    """

    PAGE = 0
    BLOCK = 1
    PARA = 2
    WORD = 3
    SYMBOL = 4

    @property
    def attr_name(self):
        name_cvt = {
            GCVFeatureType.PAGE: "pages",
            GCVFeatureType.BLOCK: "blocks",
            GCVFeatureType.PARA: "paragraphs",
            GCVFeatureType.WORD: "words",
            GCVFeatureType.SYMBOL: "symbols",
        }
        return name_cvt[self]

    @property
    def child_level(self):
        child_cvt = {
            GCVFeatureType.PAGE: GCVFeatureType.BLOCK,
            GCVFeatureType.BLOCK: GCVFeatureType.PARA,
            GCVFeatureType.PARA: GCVFeatureType.WORD,
            GCVFeatureType.WORD: GCVFeatureType.SYMBOL,
            GCVFeatureType.SYMBOL: None,
        }
        return child_cvt[self]


class GCVAgent(BaseOCRAgent):
    """A wrapper for `Google Cloud Vision (GCV) <https://cloud.google.com/vision>`_ Text
    Detection APIs.

    Note:
        Google Cloud Vision API returns the output text in two types:

        * `text_annotations`:

            In this format, GCV automatically find the best aggregation
            level for the text, and return the results in a list. We use
            :obj:`~gather_text_annotations` to reterive this type of
            information.

        * `full_text_annotation`:

            To support better user control, GCV also provides the
            `full_text_annotation` output, where it returns the hierarchical
            structure of the output text. To process this output, we provide
            the :obj:`~gather_full_text_annotation` function to aggregate the
            texts of the given aggregation level.
    """

    DEPENDENCIES = ["google-cloud-vision"]
    MODULES = [
        {"import_name": "_vision", "module_path": "google.cloud.vision"},
        {"import_name": "_json_format", "module_path": "google.protobuf.json_format"},
    ]

    def __init__(self, languages=None, ocr_image_decode_type=".png"):
        """Create a Google Cloud Vision OCR Agent.

        Args:
            languages (:obj:`list`, optional):
                You can specify the language code of the documents to detect to improve
                accuracy. The supported language and their code can be found on `this page
                <https://cloud.google.com/vision/docs/languages>`_.
                Defaults to None.

            ocr_image_decode_type (:obj:`str`, optional):
                The format to convert the input image to before sending for GCV OCR.
                Defaults to `".png"`.

                    * `".png"` is suggested as it does not compress the image.
                    * But `".jpg"` could also be a good choice if the input image is very large.
        """
        try:
            self._client = self._vision.ImageAnnotatorClient()
        except:
            warnings.warn(
                "The GCV credential has not been set. You could not run the detect command."
            )
        self._context = self._vision.types.ImageContext(language_hints=languages)
        self.ocr_image_decode_type = ocr_image_decode_type

    @classmethod
    def with_credential(cls, credential_path, **kwargs):
        """Specifiy the credential to use for the GCV OCR API.

        Args:
            credential_path (:obj:`str`): The path to the credential file
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
        return cls(**kwargs)

    def _detect(self, img_content):
        img_content = self._vision.types.Image(content=img_content)
        response = self._client.document_text_detection(
            image=img_content, image_context=self._context
        )
        return response

    def detect(
        self,
        image,
        return_response=False,
        return_only_text=False,
        agg_output_level=None,
    ):
        """Send the input image for OCR.

        Args:
            image (:obj:`np.ndarray` or :obj:`str`):
                The input image array or the name of the image file
            return_response (:obj:`bool`, optional):
                Whether directly return the google cloud response.
                Defaults to `False`.
            return_only_text (:obj:`bool`, optional):
                Whether return only the texts in the OCR results.
                Defaults to `False`.
            agg_output_level (:obj:`~GCVFeatureType`, optional):
                When set, aggregate the GCV output with respect to the
                specified aggregation level. Defaults to `None`.
        """
        if isinstance(image, np.ndarray):
            img_content = imencode(self.ocr_image_decode_type, image)[1].tostring()

        elif isinstance(image, str):
            with io.open(image, "rb") as image_file:
                img_content = image_file.read()

        res = self._detect(img_content)

        if return_response:
            return res

        if return_only_text:
            return res.full_text_annotation.text

        if agg_output_level is not None:
            return self.gather_full_text_annotation(res, agg_output_level)

        return self.gather_text_annotations(res)

    @staticmethod
    def gather_text_annotations(response):
        """Convert the text_annotations from GCV output to an :obj:`Layout` object.

        Args:
            response (:obj:`AnnotateImageResponse`):
                The returned Google Cloud Vision AnnotateImageResponse object.

        Returns:
            :obj:`Layout`: The reterived layout from the response.
        """

        # The 0th element contains all texts
        doc = response.text_annotations[1:]
        gathered_text = Layout()

        for i, text_comp in enumerate(doc):
            points = _cvt_GCV_vertices_to_points(text_comp.bounding_poly.vertices)
            gathered_text.append(
                TextBlock(block=Quadrilateral(points), text=text_comp.description, id=i)
            )

        return gathered_text

    @staticmethod
    def gather_full_text_annotation(response, agg_level):
        """Convert the full_text_annotation from GCV output to an :obj:`Layout` object.

        Args:
            response (:obj:`AnnotateImageResponse`):
                The returned Google Cloud Vision AnnotateImageResponse object.

            agg_level (:obj:`~GCVFeatureType`):
                The layout level to aggregate the text in full_text_annotation.

        Returns:
            :obj:`Layout`: The reterived layout from the response.
        """

        def iter_level(
            iter,
            agg_level=None,
            text_blocks=None,
            texts=None,
            cur_level=GCVFeatureType.PAGE,
        ):

            for item in getattr(iter, cur_level.attr_name):
                if cur_level == agg_level:
                    texts = []

                # Go down levels to fetch the texts
                if cur_level == GCVFeatureType.SYMBOL:
                    texts.append(item.text)
                elif (
                    cur_level == GCVFeatureType.WORD
                    and agg_level != GCVFeatureType.SYMBOL
                ):
                    chars = []
                    iter_level(
                        item, agg_level, text_blocks, chars, cur_level.child_level
                    )
                    texts.append("".join(chars))
                else:
                    iter_level(
                        item, agg_level, text_blocks, texts, cur_level.child_level
                    )

                if cur_level == agg_level:
                    nonlocal element_id
                    points = _cvt_GCV_vertices_to_points(item.bounding_box.vertices)
                    text_block = TextBlock(
                        block=Quadrilateral(points),
                        text=" ".join(texts),
                        score=item.confidence,
                        id=element_id,
                    )

                    text_blocks.append(text_block)
                    element_id += 1

        if agg_level == GCVFeatureType.PAGE:
            doc = response.text_annotations[0]
            points = _cvt_GCV_vertices_to_points(doc.bounding_poly.vertices)

            text_blocks = [TextBlock(block=Quadrilateral(points), text=doc.description)]

        else:
            doc = response.full_text_annotation
            text_blocks = []
            element_id = 0
            iter_level(doc, agg_level, text_blocks)

        return Layout(text_blocks)

    def load_response(self, filename):
        with open(filename, "r") as f:
            data = f.read()
        return self._json_format.Parse(
            data, self._vision.types.AnnotateImageResponse(), ignore_unknown_fields=True
        )

    def save_response(self, res, file_name):
        res = self._json_format.MessageToJson(res)

        with open(file_name, "w") as f:
            json_file = json.loads(res)
            json.dump(json_file, f)


class TesseractFeatureType(BaseOCRElementType):
    """
    The element types for Tesseract Detection API
    """

    PAGE = 0
    BLOCK = 1
    PARA = 2
    LINE = 3
    WORD = 4

    @property
    def attr_name(self):
        name_cvt = {
            TesseractFeatureType.PAGE: "page_num",
            TesseractFeatureType.BLOCK: "block_num",
            TesseractFeatureType.PARA: "par_num",
            TesseractFeatureType.LINE: "line_num",
            TesseractFeatureType.WORD: "word_num",
        }
        return name_cvt[self]

    @property
    def group_levels(self):
        levels = ["page_num", "block_num", "par_num", "line_num", "word_num"]
        return levels[: self + 1]


class TesseractAgent(BaseOCRAgent):
    """
    A wrapper for `Tesseract <https://github.com/tesseract-ocr/tesseract>`_ Text
    Detection APIs based on `PyTesseract <https://github.com/tesseract-ocr/tesseract>`_.
    """

    DEPENDENCIES = ["pytesseract"]
    MODULES = [{"import_name": "_pytesseract", "module_path": "pytesseract"}]

    def __init__(self, languages="eng", **kwargs):
        """Create a Tesseract OCR Agent.

        Args:
            languages (:obj:`list` or :obj:`str`, optional):
                You can specify the language code(s) of the documents to detect to improve
                accuracy. The supported language and their code can be found on
                `its github repo <https://github.com/tesseract-ocr/langdata>`_.
                It supports two formats: 1) you can pass in the languages code as a string
                of format like `"eng+fra"`, or 2) you can pack them as a list of strings
                `["eng", "fra"]`.
                Defaults to 'eng'.
        """
        self.lang = languages if isinstance(languages, str) else "+".join(languages)
        self.configs = kwargs

    @classmethod
    def with_tesseract_executable(cls, tesseract_cmd_path, **kwargs):

        cls._pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
        return cls(**kwargs)

    def _detect(self, img_content):
        res = {}
        res["text"] = self._pytesseract.image_to_string(
            img_content, lang=self.lang, **self.configs
        )
        _data = self._pytesseract.image_to_data(
            img_content, lang=self.lang, **self.configs
        )
        res["data"] = pd.read_csv(
            io.StringIO(_data), quoting=csv.QUOTE_NONE, encoding="utf-8", sep="\t"
        )
        return res

    def detect(
        self, image, return_response=False, return_only_text=True, agg_output_level=None
    ):
        """Send the input image for OCR.

        Args:
            image (:obj:`np.ndarray` or :obj:`str`):
                The input image array or the name of the image file
            return_response (:obj:`bool`, optional):
                Whether directly return all output (string and boxes
                info) from Tesseract.
                Defaults to `False`.
            return_only_text (:obj:`bool`, optional):
                Whether return only the texts in the OCR results.
                Defaults to `False`.
            agg_output_level (:obj:`~TesseractFeatureType`, optional):
                When set, aggregate the GCV output with respect to the
                specified aggregation level. Defaults to `None`.
        """

        res = self._detect(image)

        if return_response:
            return res

        if return_only_text:
            return res["text"]

        if agg_output_level is not None:
            return self.gather_data(res, agg_output_level)

        return res["text"]

    @staticmethod
    def gather_data(response, agg_level):
        """
        Gather the OCR'ed text, bounding boxes, and confidence
        in a given aggeragation level.
        """
        assert isinstance(
            agg_level, TesseractFeatureType
        ), f"Invalid agg_level {agg_level}"
        res = response["data"]
        df = (
            res[~res.text.isna()]
            .groupby(agg_level.group_levels)
            .apply(
                lambda gp: pd.Series(
                    [
                        gp["left"].min(),
                        gp["top"].min(),
                        gp["width"].max(),
                        gp["height"].max(),
                        gp["conf"].mean(),
                        gp["text"].str.cat(sep=" "),
                    ]
                )
            )
            .reset_index(drop=True)
            .reset_index()
            .rename(
                columns={
                    0: "x_1",
                    1: "y_1",
                    2: "w",
                    3: "h",
                    4: "score",
                    5: "text",
                    "index": "id",
                }
            )
            .assign(x_2=lambda x: x.x_1 + x.w, y_2=lambda x: x.y_1 + x.h, block_type="rectangle")
            .drop(columns=["w", "h"])
        )

        return load_dataframe(df)

    @staticmethod
    def load_response(filename):
        with open(filename, "rb") as fp:
            res = pickle.load(fp)
        return res

    @staticmethod
    def save_response(res, file_name):

        with open(file_name, "wb") as fp:
            pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)


class PaddleocrAgent():
    """
    A wrapper for `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_ Text
    Detection APIs based on `paddleocr <https://github.com/PaddlePaddle/PaddleOCR>`_.
    """

    DEPENDENCIES = ["paddleocr"]

    def __init__(self, languages="en", use_gpu=True, use_angle_cls=False, det=True, rec=True, cls=False, **kwargs):
        """Create a Tesseract OCR Agent.

        Args:
            languages (:obj:`list` or :obj:`str`, optional):
                You can specify the language code(s) of the documents to detect to improve
                accuracy. The supported language and their code can be found on
                `its github repo <https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_ch/whl.md>`_.
                It supports llaguages：`ch`, `en`, `french`, `german`, `korean`, `japan`.
                Defaults to 'eng'.
        """
        self.lang = languages
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.det = det
        self.rec = rec
        self.cls = cls
        self.configs = kwargs
        self.ocr = PaddleOCR(use_gpu=self.use_gpu, use_angle_cls=self.use_angle_cls, lang=self.lang)

    def detect(
        self, image, return_response=False, return_only_text=True, agg_output_level=None
    ):
        """Send the input image for OCR.

        Args:
            image (:obj:`np.ndarray` or :obj:`str`):
                The input image array or the name of the image file
            return_response (:obj:`bool`, optional):
                Whether directly return all output (string and boxes
                info) from Tesseract.
                Defaults to `False`.
            return_only_text (:obj:`bool`, optional):
                Whether return only the texts in the OCR results.
                Defaults to `False`.
            agg_output_level (:obj:`~TesseractFeatureType`, optional):
                When set, aggregate the GCV output with respect to the
                specified aggregation level. Defaults to `None`.
        """ 
        result = self.ocr.ocr(image, det=self.det, rec=self.rec, cls=self.cls)
        txts = '\n'.join(line[1][0] for line in result)
        return txts
