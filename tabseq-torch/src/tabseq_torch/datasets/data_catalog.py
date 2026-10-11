from pathlib import Path
import os


class DataCatalog:
    def __init__(self):
        self.home = Path(os.getenv("TABSEQ_HOME"))
        self.raw = self.home / "raw"
        self.datasets = self.home / "datasets"


DC = DataCatalog()
