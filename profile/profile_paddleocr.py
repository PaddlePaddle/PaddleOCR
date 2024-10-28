from paddleocr import PaddleOCR
from dataclasses import dataclass, asdict

from PIL.Image import Image as ImageType
import os
import numpy as np
from pdf2image import convert_from_bytes
import timeit
import cProfile


@dataclass
class PaddleSettings:
    use_gpu: bool = False
    use_xpu: bool = False
    use_mlu: bool = False
    use_angle_cls: bool = False
    enable_mkldnn: bool = True
    det: bool = True
    rec: bool = True
    cls: bool = False
    ocr_version: str = "PP-OCRv4"
    rec_batch_num: int = 6
    cpu_threads: int = 10


def _load_pdf_bytes() -> bytes:
    dir_path = os.path.dirname(__file__)
    path = os.path.join(dir_path, "resources", "sample_pdf.pdf")
    with open(path, "rb") as fh:
        return fh.read()


def pdf_to_image(pdf_bytes: bytes, dpi: int) -> ImageType:
    images = convert_from_bytes(pdf_bytes, dpi=dpi, grayscale=False)
    return images[0]


class TimeitDpi:
    def __init__(self, settings: PaddleSettings) -> None:
        self.settings = settings
        self.dpi_values = [50, 100, 150, 200, 250, 300]
        self.mean_exec_time = [0] * len(self.dpi_values)
        self.std_exec_time = [0] * len(self.dpi_values)

    def setup(self) -> None:
        self.pdf_bytes = _load_pdf_bytes()
        self.model = PaddleOCR(**asdict(self.settings))

    def timeit(self) -> None:
        self.setup()

        for idx in range(len(self.dpi_values)):
            image = pdf_to_image(self.pdf_bytes, self.dpi_values[idx])
            times = timeit.Timer(
                lambda: self.model.ocr(
                    np.asarray(image),
                    det=self.settings.det,
                    rec=self.settings.rec,
                    cls=self.settings.cls,
                )
            ).repeat(repeat=4, number=1)

            # Drop first call due to warm-up of the model.
            self.mean_exec_time[idx] = np.mean(times[1:]).item()
            self.std_exec_time[idx] = np.std(times[1:]).item()


class CprofilePaddle:
    def __init__(self, settings: PaddleSettings) -> None:
        self.settings = settings
        self.dpi_values = [50, 100, 150, 200, 250, 300]

    def setup(self) -> None:
        self.pdf_bytes = _load_pdf_bytes()
        self.model = PaddleOCR(**asdict(self.settings))

    def profile(self) -> None:
        self.setup()

        for dpi in self.dpi_values:
            image = pdf_to_image(self.pdf_bytes, dpi)
            with cProfile.Profile() as pr:
                self.model.ocr(
                    np.asarray(image),
                    det=self.settings.det,
                    rec=self.settings.rec,
                    cls=self.settings.cls,
                )
            os.makedirs("cprofile", exist_ok=True)
            pr.dump_stats(f"cprofile/profile_paddle_{dpi}.prof")


def print_recognition_pred():
    pdf_bytes = _load_pdf_bytes()
    model = PaddleOCR(det=False, rec=True, cls=False)
    image = pdf_to_image(pdf_bytes, dpi=300)
    ocr_result = model.ocr(np.asarray(image), det=False, rec=True, cls=False)
    print(ocr_result)


if __name__ == "__main__":
    # print_recognition_pred()

    settings = PaddleSettings(
        det=True, rec=True, cls=False, rec_batch_num=6, cpu_threads=6
    )
    timeit_dpi = TimeitDpi(settings)
    timeit_dpi.dpi_values = [200]
    timeit_dpi.timeit()
    for idx in range(len(timeit_dpi.dpi_values)):
        print(
            f"DPI: {timeit_dpi.dpi_values[idx]} | "
            f"{timeit_dpi.mean_exec_time[idx]:.4f} +/- "
            f"{timeit_dpi.std_exec_time[idx]:.4f}"
        )

    # settings = PaddleSettings(det=True, rec=True, cls=False)
    # cprofile_paddle = CprofilePaddle(settings)
    # cprofile_paddle.profile()
