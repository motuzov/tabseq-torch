from pathlib import Path


class DataCatalog:
    """
    Paths to the data (payload and meta) prepared for loading into the LazyDataset
    """

    def __init__(
        self,
        cat: Path,
        num: Path | None = None,
        targets: Path | None = None,
    ):
        self._cat: Path = cat
        self.num = num
        self.targets: Path | None = targets

    def cat2code_json(self, colset_name: str) -> Path:
        return self._cat / colset_name / "meta/cat2code.json"

    def catnum_json(self, colset_name: str) -> Path:
        return self._cat / colset_name / "meta/catnum.json"

    def cat(self, colset_name: str = "all") -> Path:
        # payload
        return self._cat / colset_name / "tb"

    def print_colsets(self) -> None:
        for setname_dir in self._cat.glob("*/meta"):
            print(setname_dir.name)
