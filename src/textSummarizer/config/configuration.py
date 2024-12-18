from textSummarizer.constants import (CONFIG_FILE_PATH,
                                      PARAMS_FILE_PATH)
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity import (DataIngestionConfig,
                                   DataValidationConfiguration,
                                   DataTransformConfiguration,
                                   ModelTrainingConfiguration)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfiguration:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfiguration(
            root_dir=config.root_dir,
            Status_file=config.Status_file,
            Required_files=config.Required_files
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformConfiguration:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformConfiguration(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name
        )
        return data_transformation_config

    def get_model_training_config(self) -> ModelTrainingConfiguration:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfiguration(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt=config.model_ckpt,
            num_train_epochs=params.num_train_epochs,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            evaluation_strategy=params.evaluation_strategy,
            eval_steps=params.eval_steps,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps
        )
        return model_training_config
