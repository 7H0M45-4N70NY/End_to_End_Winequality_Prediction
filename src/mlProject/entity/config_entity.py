from dataclasses import dataclass
from pathlib import Path

#Return type of data ingestion process
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir:Path
    train_data_path:Path
    test_data_path:Path
    model_name:str
    target_column:str
    n_estimators:float
    
@dataclass(frozen=True)
class ModelEvalConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name : Path
    target_column: str