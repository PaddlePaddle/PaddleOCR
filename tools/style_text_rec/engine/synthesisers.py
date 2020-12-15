import os

from utils.config import ArgsParser, load_config, override_config
from utils.logging import get_logger
from engine import style_samplers, corpus_generators, text_drawers, predictors, writers


class ImageSynthesiser(object):
    def __init__(self):
        self.FLAGS = ArgsParser().parse_args()
        self.config = load_config(self.FLAGS.config)
        self.config = override_config(self.config, options=self.FLAGS.override)
        self.output_dir = self.config["Global"]["output_dir"]
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.logger = get_logger(
            log_file='{}/predict.log'.format(self.output_dir))

        self.text_drawer = text_drawers.StdTextDrawer(self.config)

        predictor_method = self.config["Predictor"]["method"]
        assert predictor_method is not None
        self.predictor = getattr(predictors, predictor_method)(self.config)

    def synth_image(self, corpus, style_input, language="en"):
        corpus, text_input = self.text_drawer.draw_text(corpus, language)
        synth_result = self.predictor.predict(style_input, text_input)
        return synth_result


class DatasetSynthesiser(ImageSynthesiser):
    def __init__(self):
        super(DatasetSynthesiser, self).__init__()
        self.tag = self.FLAGS.tag
        self.output_num = self.config["Global"]["output_num"]
        corpus_generator_method = self.config["CorpusGenerator"]["method"]
        self.corpus_generator = getattr(corpus_generators,
                                        corpus_generator_method)(self.config)

        style_sampler_method = self.config["StyleSampler"]["method"]
        assert style_sampler_method is not None
        self.style_sampler = style_samplers.DatasetSampler(self.config)
        self.writer = writers.SimpleWriter(self.config, self.tag)

    def synth_dataset(self):
        for i in range(self.output_num):
            style_data = self.style_sampler.sample()
            style_input = style_data["image"]
            corpus_language, text_input_label = self.corpus_generator.generate(
            )
            text_input_label, text_input = self.text_drawer.draw_text(
                text_input_label, corpus_language)

            synth_result = self.predictor.predict(style_input, text_input)
            fake_fusion = synth_result["fake_fusion"]
            self.writer.save_image(fake_fusion, text_input_label)
        self.writer.save_label()
        self.writer.merge_label()
