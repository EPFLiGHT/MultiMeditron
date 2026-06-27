import json
import warnings

class JSONLGenerator:
    """Lazy line-by-line iterator over a JSONL file.

    Each call to ``next()`` reads one line and parses it as JSON.
    Invalid lines emit a warning and yield ``None``.

    Args:
        file_path (str): Path to the ``.jsonl`` file.
    """

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
        """Reopen the file to restart iteration from the beginning."""
        self.file = open(self.file_path, 'r')



