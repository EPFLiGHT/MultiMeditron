from multimeditron.dataset.preprocessor import BaseDatasetPreprocessor, AutoDatasetPreprocessor    

@AutoDatasetPreprocessor.register("python")
class PythonProcessor(BaseDatasetPreprocessor):
    """Preprocessor that applies a user-defined Python expression to each sample via ``Dataset.map``."""

    def _process(self, ds, num_processes, func=None, imports=[], remove_columns=[]):
        _exec_imports(imports)

        def fn_func(data, idx):
            return _exec_py(idx, data, func)

        return ds.map(
            fn_func,
            batched=False,
            # writer_batch_size=num_processes,
            num_proc=num_processes,
            with_indices=True,
            remove_columns=remove_columns,
        )

@AutoDatasetPreprocessor.register("python-filter")
class PythonFilterProcessor(BaseDatasetPreprocessor):
    """Preprocessor that filters samples using a user-defined Python predicate via ``Dataset.filter``."""

    def _process(self, ds, num_processes, func=None, imports=[]):
        _exec_imports(imports)

        def fn_func(data, idx):
            return _exec_py(idx, data, func)

        return ds.filter(
            fn_func,
            batched=False,
            # writer_batch_size=num_processes,
            num_proc=num_processes,
            with_indices=True
        )

def _exec_imports(imports):
    """Import the named modules and bind them into this module's globals.

    Makes the modules available to the user code run by :func:`_exec_py`.

    Args:
        imports (list[str] | None): Module names to import (e.g. ``["re", "json"]``).
    """
    if imports is not None and len(imports) > 0:
        import importlib
        for imp in imports:
            globals()[imp] = importlib.import_module(imp)

def _exec_py(idx, data, code):
    """Evaluate user-provided Python against one dataset sample.

    The config supplies ``code`` as either a single expression string or a list
    of statements whose final entry is the expression to return. It is run with
    ``eval``/``exec`` and therefore trusted — only use configs you control.

    Args:
        idx (int): Sample index (available to the user code as ``idx``).
        data (dict): The sample (available to the user code as ``data``).
        code (str | list[str]): Expression, or statements ending in an expression.

    Returns:
        The value of the final evaluated expression (the transformed/predicate result).
    """
    if isinstance(code, str):
        return eval(code, globals(), locals())
    
    elif len(code) > 1:
        # exec everything except the last line
        for line in code[:-1]:
            exec(line, globals(), locals())

        # eval the last line
        return eval(code[-1], globals(), locals())