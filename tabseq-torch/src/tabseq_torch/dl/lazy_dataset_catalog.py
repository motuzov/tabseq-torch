from pathlib import Path


class DataCatalog:
    """
    The catalog 
    """
    def __init__(
        self,
        src_tabts_path: Path,
        cat_path: Path,
        meta_path: Path,
        targets_path: Path | None = None,
    ):
        self.src: Path = src_tabts_path
        self._cat: Path = cat_path
        self._meta: Path = meta_path
        self.targets: Path | None = targets_path

    def cat2code_json(self, setname: str) -> Path:
        return self._meta / setname / "meta/cat2code.json"

    def catnum_json(self, setname: str) -> Path:
        return self._meta / setname / "meta/catnum.json"

    def cat(self, setname: str = "") -> Path:
        # payload
        if setname:
            return self._cat / setname / "tb"
        return self._cat

    def print_setnames(self) -> None:
        for setname_dir in self._meta.iterdir():
            print(setname_dir.name)
