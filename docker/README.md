# Container & reproducible environment

The container image is the **canonical reproducible environment** for MultiMeditron.
All training and evaluation on CSCS Clariden runs inside it. Use it (rather than a
bare `pip install`) whenever you need results to match.

## What's here

| File | Purpose |
|---|---|
| `Dockerfile` | Build recipe for the image (base: `nvcr.io/nvidia/pytorch:25.05-py3`). |
| `Dockerfile.verl` | Variant image including the verl RL stack. |
| `entrypoint.sh` | Drops container root to the invoking user's UID/GID. |

The CSCS **EDF** (Environment Definition File) the SLURM launchers use lives at
[`../cookbook/assets/edf.toml`](../cookbook/assets/edf.toml).

## CSCS Clariden (SLURM + EDF)

The `sbatch_*.sh` launchers run with `--environment=~/.edf/multimeditron.toml`, so
install the in-repo EDF to that location once:

```bash
mkdir -p ~/.edf
cp cookbook/assets/edf.toml ~/.edf/multimeditron.toml
```

The EDF references a prebuilt image and sets the Slingshot / AWS-libfabric NCCL
configuration required on GH200 nodes — including **`NCCL_NET_GDR_LEVEL = "0"`**,
without which multi-node training hangs on the first NCCL collective (see
[`../docs/source/guides/troubleshooting.rst`](../docs/source/guides/troubleshooting.rst)).
The `/capstor`, `/iopsstor` and `/users` filesystems are mounted into the container.

## Local / non-CSCS use

Pull the prebuilt image:

```bash
docker pull michelducartier24/multimeditron-git:latest-arm64   # GH200 / ARM64
docker pull michelducartier24/multimeditron-git:latest-amd64   # x86_64
```

Or rebuild it from source:

```bash
docker build -f docker/Dockerfile -t multimeditron:local .
```

The build installs the package (`pip install -e ".[flash-attn]"`) and the
`EPFLiGHT/lmms-eval` evaluation harness.

## Reproducibility notes

- **Pin the image for the submission.** The EDF uses the mutable tag
  `latest-arm64`. For an exact, reproducible record, resolve and pin the image
  **digest** instead of `latest`:

  ```bash
  docker buildx imagetools inspect michelducartier24/multimeditron-git:latest-arm64
  # then set image = "michelducartier24/multimeditron-git@sha256:<digest>" in the EDF
  ```

- **Exact package versions** are captured in
  [`../requirements-lock.txt`](../requirements-lock.txt) (a `pip freeze` from inside
  this container). For a non-container install that matches the container:

  ```bash
  pip install -r requirements-lock.txt
  ```

  Note: entries pinned as `@ file:///opt/...` (torch, torchvision, flash_attn,
  torch_tensorrt) ship in the NGC base image and are not on PyPI — use the
  container for those. Regenerate the lock after changing dependencies:

  ```bash
  srun -A a127 --partition=debug --time=00:03:00 --nodes=1 \
    --environment=$HOME/.edf/multimeditron.toml bash -lc 'pip freeze' > requirements-lock.txt
  ```
