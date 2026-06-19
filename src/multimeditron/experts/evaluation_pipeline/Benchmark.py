from abc import ABC, abstractmethod


# abstract class from which all CLIP expert benchmarks must inherit
class Benchmark(ABC):

    # takes as input input the path of the evaluated model and returns the evaluation metrics which has to be maximized
    @abstractmethod
    def evaluate(self, model_path):
        pass
