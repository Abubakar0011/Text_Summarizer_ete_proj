import os
# from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import DataTransformConfiguration


class DataTransform:
    def __init__(self, config: DataTransformConfiguration):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name)
       
    def convert_examples_to_features(self, examples_batch):
        inputs_encoding = self.tokenizer(examples_batch['dialogue'], 
                                         max_length=1024, truncation=True)
        
        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(examples_batch['summary'],
                                             max_length=128, truncation=True)
            
        return {
            'input_ids': inputs_encoding['input_ids'],
            'attention_mask': inputs_encoding['attention_mask'],
            'labels': target_encoding['input_ids']
        }

    def convert(self):
        dataet_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataet_samsum.map(
            self.convert_examples_to_features,
            batched=True)
        dataset_samsum_pt.save_to_disk(os.path.join(
            self.config.root_dir, "samsum_dataset"))
