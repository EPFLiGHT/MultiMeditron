def python_processor(ds, num_processes, func=None):
    def process_fn(data, idx):
        return eval(func, globals(), locals())

    return ds.map(
        process_fn,
        batched=False,
        writer_batch_size=num_processes,
        num_proc=num_processes,
        with_indices=True
    )