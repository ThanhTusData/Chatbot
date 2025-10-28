import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path: str = 'config/logging.yaml', default_level=logging.INFO):
    path = Path(config_path)
    
    if path.exists():
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)