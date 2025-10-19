from multimeditron.dataset.preprocessor import BaseDatasetPreprocessor, AutoDatasetPreprocessor    

@AutoDatasetPreprocessor.register("shuffle")
class ShuffleProcessor(BaseDatasetPreprocessor):
    def _process(self, ds, num_processes, seed=42):
        return ds.shuffle(seed=seed)