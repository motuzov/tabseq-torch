import os
from pathlib import Path

from load_dotenv import load_dotenv

load_dotenv()


class DataCatalog:
    def __init__(self):
        self.home = Path(os.getenv("TABSEQ_HOME"))
        self.raw = self.home / "raw"
        self.datasets = self.home / "datasets"


DC = DataCatalog()
