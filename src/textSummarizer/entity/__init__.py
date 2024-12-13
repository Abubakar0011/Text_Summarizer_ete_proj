from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfiguration:
    root_dir: Path
    Status_file: str
    Required_files: list


@dataclass(frozen=True)
class DataTransformConfiguration:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path
