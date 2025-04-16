import os
from enum import Enum


# Data types ###
class DataType(Enum):
    TEXT = 1
    PDF_DIR = 2


# Data class ###
class Data:
    def __init__(
        self,
        path,
        data_type: DataType,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.path = path
        self.data_type = data_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def __eq__(self, other) -> bool:
        return all((
            isinstance(other, Data),
            self.path == other.path,
            self.data_type == other.data_type,
            self.chunk_size == other.chunk_size,
            self.chunk_overlap == other.chunk_overlap,
        ))

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)


class DataList:
    def __init__(self, config: dict):
        self.data = []
        self.config = config
        self.main_dir = config["paths"]["data"]

    def get_data_type(self, path) -> DataType | None:
        """
        Get the type of the data

        Args:
            path (str): Path of the data

        Returns:
            DataType: Type of the data
        """
        if path is None:
            return None

        suffix_lookup = {
            ".txt": DataType.TEXT,
        }
        for suffix, datatype in suffix_lookup.items():
            if path.endswith(suffix):
                return datatype
        
        if os.path.isdir(path) and any(f.endswith('.pdf') for f in os.listdir(path)):
            return DataType.PDF_DIR

        return None

    def add(self, path, chunk_size=0, chunk_overlap=0) -> None:
        """
        Add a single data file

        Args:
            path (str): Path of the file
        """
        path = self.main_dir + path
        data_type = self.get_data_type(path)
        if data_type is None:
            print(f"\33[1;31m[DataTester]\33[0m: Il path {path} non è valido")
            return
        d = Data(path, data_type, chunk_size, chunk_overlap)
        self.data.append(d)

    def add_dir(self, path="", chunk_size=0, chunk_overlap=0) -> None:
        """
        Add all the files in a directory

        Args:
            path (str): Path of the directory
        """
        for file in os.listdir(self.main_dir + path):
            self.add(path + file, chunk_size, chunk_overlap)

    def test(self) -> bool:
        """
        Check if data is valid

        Returns:
            bool: True if data is valid, False otherwise
        """
        if not self.data:
            print("\33[1;31m[DataTester]\33[0m: Nessun dato presente")
            return False
        for d in self.data:
            if not os.path.exists(d.path):
                print(f"\33[1;31m[DataTester]\33[0m: Il file {d.path} non esiste")
                return False

        print("\33[1;32m[DataTester]\33[0m: Dati validati con successo")
        return True

    def print_data(self) -> None:
        """
        Print the data
        """
        for d in self.data:
            print(
                f"Path: {d.path}, Type: {d.data_type}, Chunk Size: {d.chunk_size}, Chunk Overlap: {d.chunk_overlap}"
            )

    def get_data(self) -> list:
        """
        Get the data

        Returns:
            list: List of data
        """
        return self.data
