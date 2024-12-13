from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.data_transformation import DataTransform
# from textSummarizer.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transf_config = config.get_data_transformation_config()
        data_transf = DataTransform(config=data_transf_config)
        data_transf.convert()
