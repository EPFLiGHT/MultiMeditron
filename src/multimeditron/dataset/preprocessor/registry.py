from multimeditron.dataset.preprocessor.python import python_processor
import logging

logger = logging.getLogger(__name__)


_MULTIMEDITRON_DATASET_PREPROCESSOR_REGISTRY = {
    "python": python_processor,
}

def run_processors(ds, num_processes, processors):
    from datasets import enable_caching, disable_caching, is_caching_enabled

    # Disable caching as it often causes desync issues
    was_caching_enabled = is_caching_enabled()
    disable_caching()

    # Run each processor in sequence
    for proc in processors:
        logger.info(f"Running processor: {proc.type} with args: {proc.kwargs}")

        if proc.type not in _MULTIMEDITRON_DATASET_PREPROCESSOR_REGISTRY:
            raise ValueError(
                f"Processor type {proc.type} not recognized. Available types: {list(_MULTIMEDITRON_DATASET_PREPROCESSOR_REGISTRY.keys())}"
            )

        processor_fn = _MULTIMEDITRON_DATASET_PREPROCESSOR_REGISTRY[proc.type]
        ds = processor_fn(ds, num_processes, **proc.kwargs)

    # Restore previous caching state
    if was_caching_enabled:
        enable_caching()
    return ds