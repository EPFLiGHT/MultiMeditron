import json
import warnings

class JSONLGenerator:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file = open(file_path, 'r', encoding='utf-8')

    def __iter__(self):
        return self

    def __next__(self):
        line = self.file.readline()
        if not line:
            raise StopIteration

        try:
            d = json.loads(line)
        except:
            warnings.warn(f"JSON error: Can't load {line}")
            return None

        return d

    def __del__(self):
        self.file.close()

    def reset(self):
        self.file = open(self.file_path, 'r')



