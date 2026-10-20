import os
from pathlib import Path

from load_dotenv import load_dotenv

load_dotenv()


class TrainingCatalog:
    def __init__(self, dataset_name: str):
        if "TABSEQ_HOME" not in os.environ:
            raise ValueError("Environment variable TABSEQ_HOME doesn't exist")
        self.home = Path(os.getenv("TABSEQ_HOME"))
        self.training = self.home / "training"
        self.tb_logs = self.training / dataset_name / "tb_logs"
