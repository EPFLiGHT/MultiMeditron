from abc import ABC, abstractmethod


class Benchmark(ABC):

    # takes as input input the path of the evaluated model and returns the evaluation metrics which has to be maximized
    @abstractmethod
    def evaluate(self, model_path):
        pass
