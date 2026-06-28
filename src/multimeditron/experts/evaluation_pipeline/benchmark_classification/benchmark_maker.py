#!/usr/bin/env python3
"""Generate a ClassificationBenchmark subclass from a YAML manifest.

Usage:
    python benchmark_maker.py manifest.yaml
    python benchmark_maker.py manifest.yaml --output /path/to/output.py
    python benchmark_maker.py manifest.yaml --print
"""

import argparse
import sys
import textwrap
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def _to_class_name(name: str) -> str:
    return "".join(part.title() for part in name.replace("-", "_").split("_")) + "Benchmark"


def _format_set_literal(items) -> str:
    quoted = ", ".join(f'"{item}"' for item in sorted(items))
    return "{" + quoted + "}"


def _cache_infix(allowed_subdatasets, stratify_by_label):
    parts = []
    if allowed_subdatasets:
        parts.append("clean")
    if stratify_by_label:
        parts.append("strat")
    return ("_" + "_".join(parts)) if parts else ""


def generate(manifest: dict) -> str:
    name = manifest["name"]
    num_classes = manifest["num_classes"]
    manifest_subdir = manifest["manifest_subdir"]
    class_name = manifest.get("class_name") or _to_class_name(name)

    labels = manifest.get("labels")
    max_train = manifest.get("max_train_examples")
    max_test = manifest.get("max_test_examples")
    allowed_subdatasets = manifest.get("allowed_subdatasets")
    stratify = manifest.get("stratify_by_label", False)
    add_is_available = manifest.get("is_available", False)
    seed_train = manifest.get("seed_train", 42)
    seed_test = manifest.get("seed_test", 43)
    docstring = manifest.get("docstring", "")

    infix = _cache_infix(allowed_subdatasets, stratify)
    train_prefix = f"{name}{infix}_mlp_train"
    test_prefix = f"{name}{infix}_benchmark_eval"

    lines = []

    lines.append("from pathlib import Path")
    lines.append("")
    lines.append("from .base import ClassificationBenchmark")
    lines.append("from .multimediset_manifest import DEFAULT_MANIFEST_ROOT, load_or_build_manifest_dataset")

    if allowed_subdatasets:
        lines.append("")
        lines.append("")
        lines.append(f"CLEAN_SUBDATASETS = {_format_set_literal(allowed_subdatasets)}")

    lines.append("")
    lines.append("")
    lines.append(f"class {class_name}(ClassificationBenchmark):")

    if docstring:
        lines.append(f'    """{docstring}"""')
        lines.append("")

    lines.append(f'    name = "{name}"')
    lines.append(f"    num_classes = {num_classes}")
    lines.append(f'    default_manifest_root = DEFAULT_MANIFEST_ROOT / "{manifest_subdir}"')

    if labels:
        lines.append("    labels = [")
        for label in labels:
            lines.append(f'        "{label}",')
        lines.append("    ]")
        lines.append("    label_to_idx = {label: idx for idx, label in enumerate(labels)}")

    if max_train is not None:
        lines.append(f"    max_train_examples = {max_train:_}")
    if max_test is not None:
        lines.append(f"    max_test_examples = {max_test:_}")

    lines.append("")
    lines.append("    def __init__(")
    lines.append("        self,")
    lines.append("        cache_root=None,")
    lines.append("        max_train_examples=None,")
    lines.append("        max_test_examples=None,")
    lines.append("        manifest_root=None,")
    lines.append("    ):")
    lines.append("        super().__init__(")
    lines.append("            cache_root=cache_root,")
    lines.append("            max_train_examples=max_train_examples,")
    lines.append("            max_test_examples=max_test_examples,")
    lines.append("        )")
    lines.append("        self.manifest_root = (")
    lines.append(
        "            Path(manifest_root) if manifest_root is not None else self.default_manifest_root"
    )
    lines.append("        )")

    if add_is_available:
        lines.append("")
        lines.append("    @classmethod")
        lines.append("    def is_available(cls, manifest_root=None):")
        lines.append("        root = (")
        lines.append(
            "            Path(manifest_root) if manifest_root is not None else cls.default_manifest_root"
        )
        lines.append("        )")
        lines.append(
            '        return (root / "mlp_train.jsonl").exists() and (root / "benchmark_eval.jsonl").exists()'
        )

    lines.append("")
    lines.append("    def build_train_dataset(self, model, model_name, use_cache=True):")
    lines.append("        return load_or_build_manifest_dataset(")
    lines.append('            manifest_path=self.manifest_root / "mlp_train.jsonl",')
    lines.append(f'            cache_prefix=f"{{model_name}}_{train_prefix}",')
    lines.append("            model=model,")
    lines.append("            cache_root=self.cache_root,")
    lines.append("            use_cache=use_cache,")
    lines.append(f'            desc="{name}-mlp-train",')
    lines.append("            max_examples=self.max_train_examples,")
    lines.append(f"            seed={seed_train},")
    if allowed_subdatasets:
        lines.append("            allowed_subdatasets=CLEAN_SUBDATASETS,")
    if stratify:
        lines.append("            stratify_by_label=True,")
    lines.append("        )")

    lines.append("")
    lines.append("    def build_test_dataset(self, model, model_name, use_cache=True):")
    lines.append("        return load_or_build_manifest_dataset(")
    lines.append('            manifest_path=self.manifest_root / "benchmark_eval.jsonl",')
    lines.append(f'            cache_prefix=f"{{model_name}}_{test_prefix}",')
    lines.append("            model=model,")
    lines.append("            cache_root=self.cache_root,")
    lines.append("            use_cache=use_cache,")
    lines.append(f'            desc="{name}-benchmark-eval",')
    lines.append("            max_examples=self.max_test_examples,")
    lines.append(f"            seed={seed_test},")
    if allowed_subdatasets:
        lines.append("            allowed_subdatasets=CLEAN_SUBDATASETS,")
    if stratify:
        lines.append("            stratify_by_label=True,")
    lines.append("        )")

    return "\n".join(lines) + "\n"


def _print_next_steps(manifest: dict) -> None:
    name = manifest["name"]
    class_name = manifest.get("class_name") or _to_class_name(name)
    NAME_UPPER = name.upper()

    print()
    print("Next steps:")
    print(f"  1. benchmark_classification/__init__.py — add:")
    print(f"       from .{name}_benchmark import {class_name}")
    print(f"       # and '{class_name}' to __all__")
    print()
    print(f"  2. build_benchmarks.py — add env-var dict and builder, e.g.:")
    print(f"       {NAME_UPPER}_ENV_VARS = {{")
    print(f'           "max_train_examples": "{NAME_UPPER}_MAX_TRAIN_EXAMPLES",')
    print(f'           "max_test_examples":  "{NAME_UPPER}_MAX_TEST_EXAMPLES",')
    print(f"       }}")
    print()
    print(f"       def _maybe_build_{name}_benchmark():")
    print(f"           if not _manifest_pair_exists({class_name}.default_manifest_root):")
    print(f"               return None")
    print(f"           return {class_name}(")
    print(
        f"               max_train_examples=_parse_optional_int(os.environ.get({NAME_UPPER}_ENV_VARS[\"max_train_examples\"])),"
    )
    print(
        f"               max_test_examples=_parse_optional_int(os.environ.get({NAME_UPPER}_ENV_VARS[\"max_test_examples\"])),"
    )
    print(f"           )")
    print()
    print(f"     Then add '{name}' to available_builders in build_benchmarks_from_names().")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a ClassificationBenchmark subclass from a YAML manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            After generation, register the new benchmark:
              1. Add the import to benchmark_classification/__init__.py
              2. Add the builder and env-var dict to build_benchmarks.py
              3. Add the name to available_builders in build_benchmarks_from_names()
            """
        ),
    )
    parser.add_argument("manifest", help="Path to the YAML manifest file.")
    parser.add_argument(
        "--output",
        "-o",
        help="Output .py file path. Defaults to benchmark_classification/<name>_benchmark.py.",
    )
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print generated code to stdout without writing a file.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing file without prompting.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)

    for field in ("name", "num_classes", "manifest_subdir"):
        if field not in manifest:
            print(f"Error: manifest missing required field: {field!r}", file=sys.stderr)
            sys.exit(1)

    code = generate(manifest)

    if args.print_only:
        print(code)
        return

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(__file__).parent / f"{manifest['name']}_benchmark.py"

    if out_path.exists() and not args.force:
        print(f"Warning: {out_path} already exists. Overwrite? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

    out_path.write_text(code, encoding="utf-8")
    print(f"Generated: {out_path}")
    _print_next_steps(manifest)


if __name__ == "__main__":
    main()
