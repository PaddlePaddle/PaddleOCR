import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_gen_label():
    path = Path(__file__).resolve().parents[1] / "ppocr" / "utils" / "gen_label.py"
    spec = importlib.util.spec_from_file_location("gen_label", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DetLabelTest(unittest.TestCase):
    def test_a_comma_inside_the_transcription_is_kept(self):
        gen_label = load_gen_label()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels"
            labels.mkdir()
            (labels / "gt_img_1.txt").write_text(
                "0,0,10,0,10,10,0,10,Theatre\n"
                "0,0,10,0,10,10,0,10,Hello, world\n",
                encoding="utf-8",
            )
            out = root / "out.txt"
            gen_label.gen_det_label(str(root / "imgs"), str(labels), str(out))
            line = out.read_text(encoding="utf-8").splitlines()[0]
            payload = json.loads(line.split("\t", 1)[1])
        texts = [item["transcription"] for item in payload]
        self.assertEqual(texts, ["Theatre", "Hello, world"])


if __name__ == "__main__":
    unittest.main()
