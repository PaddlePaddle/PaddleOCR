from iopath.common.file_io import PathHandler, PathManager, HTTPURLHandler
from iopath.common.file_io import PathManager as PathManagerBase

# A trick learned from https://github.com/facebookresearch/detectron2/blob/65faeb4779e4c142484deeece18dc958c5c9ad18/detectron2/utils/file_io.py#L3

MODEL_CATALOG = {
    "HJDataset": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/6icw6at8m28a2ho/model_final.pth?dl=1",
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/893paxpy5suvlx9/model_final.pth?dl=1",
        "retinanet_R_50_FPN_3x": "https://www.dropbox.com/s/yxsloxu3djt456i/model_final.pth?dl=1",
    },
    "PubLayNet": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/dgy9c10wykk4lq4/model_final.pth?dl=1",
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/d9fc9tahfzyl6df/model_final.pth?dl=1",
        "mask_rcnn_X_101_32x8d_FPN_3x": "https://www.dropbox.com/s/57zjbwv6gh3srry/model_final.pth?dl=1",
    },
    "PrimaLayout": {
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/h7th27jfv19rxiy/model_final.pth?dl=1"
    },
    "NewspaperNavigator": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/6ewh6g8rqt2ev3a/model_final.pth?dl=1",
    },
    "TableBank": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/8v4uqmz1at9v72a/model_final.pth?dl=1",
        "faster_rcnn_R_101_FPN_3x": "https://www.dropbox.com/s/6vzfk8lk9xvyitg/model_final.pth?dl=1",
    },
}

CONFIG_CATALOG = {
    "HJDataset": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/j4yseny2u0hn22r/config.yml?dl=1",
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/4jmr3xanmxmjcf8/config.yml?dl=1",
        "retinanet_R_50_FPN_3x": "https://www.dropbox.com/s/z8a8ywozuyc5c2x/config.yml?dl=1",
    },
    "PubLayNet": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/f3b12qc4hc0yh4m/config.yml?dl=1",
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/u9wbsfwz4y0ziki/config.yml?dl=1",
        "mask_rcnn_X_101_32x8d_FPN_3x": "https://www.dropbox.com/s/nau5ut6zgthunil/config.yaml?dl=1",
        "ppyolov2_r50vd_dcn_365e_publaynet": "https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar",
    },
    "PrimaLayout": {
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/yc92x97k50abynt/config.yaml?dl=1"
    },
    "NewspaperNavigator": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/wnido8pk4oubyzr/config.yml?dl=1",
    },
    "TableBank": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/7cqle02do7ah7k4/config.yaml?dl=1",
        "faster_rcnn_R_101_FPN_3x": "https://www.dropbox.com/s/h63n6nv51kfl923/config.yaml?dl=1",
        "ppyolov2_r50vd_dcn_365e_tableBank_word": "https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_word.tar",
        "ppyolov2_r50vd_dcn_365e_tableBank_latex": "https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_latex.tar",
    },
}

LABEL_MAP_CATALOG = {
    "HJDataset": {
        1: "Page Frame",
        2: "Row",
        3: "Title Region",
        4: "Text Region",
        5: "Title",
        6: "Subtitle",
        7: "Other",
    },
    "PubLayNet": {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    "PrimaLayout": {
        1: "TextRegion",
        2: "ImageRegion",
        3: "TableRegion",
        4: "MathsRegion",
        5: "SeparatorRegion",
        6: "OtherRegion",
    },
    "NewspaperNavigator": {
        0: "Photograph",
        1: "Illustration",
        2: "Map",
        3: "Comics/Cartoon",
        4: "Editorial Cartoon",
        5: "Headline",
        6: "Advertisement",
    },
    "TableBank": {0: "Table"},
}


class DropboxHandler(HTTPURLHandler):
    """
    Supports download and file check for dropbox links
    """

    def _get_supported_prefixes(self):
        return ["https://www.dropbox.com"]

    def _isfile(self, path):
        return path in self.cache_map


class LayoutParserHandler(PathHandler):
    """
    Resolve anything that's in LayoutParser model zoo.
    """

    PREFIX = "lp://"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path, **kwargs):
        model_name = path[len(self.PREFIX) :]
        dataset_name, *model_name, data_type = model_name.split("/")

        if data_type == "weight":
            model_url = MODEL_CATALOG[dataset_name]["/".join(model_name)]
        elif data_type == "config":
            model_url = CONFIG_CATALOG[dataset_name]["/".join(model_name)]
        else:
            raise ValueError(f"Unknown data_type {data_type}")
        return PathManager.get_local_path(model_url, **kwargs)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager = PathManagerBase()
PathManager.register_handler(DropboxHandler())
PathManager.register_handler(LayoutParserHandler())
