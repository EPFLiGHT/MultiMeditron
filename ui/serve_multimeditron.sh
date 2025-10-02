#!/bin/bash
#SBATCH --job-name serve-multimeditron
#SBATCH --output /users/theoschiff/meditron/reports/ui/R-%x.%j.out
#SBATCH --error  /users/theoschiff/meditron/reports/ui/R-%x.%j.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1              # 1 app process
#SBATCH --gres gpu:4                   # request 4 GPUs (the app will auto-shard if >1)
#SBATCH --cpus-per-task 16
#SBATCH --time 1:00:00
#SBATCH --environment /users/theoschiff/.edf/multimodal.toml
#SBATCH -A a127

echo "START TIME: $(date)"
set -eo pipefail
set -x

######################
### Environment    ###
######################

export HF_TOKEN="${HF_TOKEN}"
export HF_HOME=/capstor/store/cscs/swissai/a127/homes/theoschiff/hf_home

export CUDA_LAUNCH_BLOCKING=0


######################
### App settings   ###
######################
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"         # -> .../meditron/MultiMeditron
APP_REL="ui/app.py"                                # app lives in ui/
MODEL_DIR="/capstor/store/cscs/swissai/a127/homes/theoschiff/models/MultiMeditron-8B-Clip/checkpoint-813"
BASE_PATH="/capstor/store/cscs/swissai/a127/homes/theoschiff"       # where our images live for FileSystemImageRegistry
PORT=49160
HOST="0.0.0.0"

######################
# SSL / networking fixes for httpx/Gradio
######################

unset SSL_CERT_FILE
unset SSL_CERT_DIR
unset REQUESTS_CA_BUNDLE
unset CURL_CA_BUNDLE

# set SSL_CERT_FILE to a valid CA bundle path - fix for httpx failures 
if [[ -f /etc/ssl/certs/ca-certificates.crt ]]; then
  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
elif [[ -f /etc/pki/tls/certs/ca-bundle.crt ]]; then
  export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
fi


# export CUDA_VISIBLE_DEVICES=0

######################
# Install user-space deps
######################
python - <<'PY'
import importlib, sys, subprocess
pkgs = {
  "gradio": "gradio>=4.0,<5",
  "httpx": "httpx>=0.25,<0.28",
}
missing = []
for mod, spec in pkgs.items():
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(spec)
if missing:
    print("Installing:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--upgrade", "--no-cache-dir", *missing])
else:
    print("All deps present")
PY

######################
### Launch         ###
######################

SRUN_ARGS=" \
  --cpus-per-task ${SLURM_CPUS_PER_TASK} \
  --cpu-bind=none
  --jobid ${SLURM_JOB_ID} \
  --wait 60 \
  -A a127 \
  --reservation=sai-a127 \
"

APP_CMD="cd '${REPO_DIR}' && \
python '${APP_REL}' \
  --model_checkpoint '${MODEL_DIR}' \
  --base_path '${BASE_PATH}' \
  --server_name ${HOST} \
  --server_port ${PORT}
"

echo "${APP_CMD}"

echo "To access the app, after app initialisation, run:"
echo "ssh -J user@cluster -N -L ${PORT}:localhost:${PORT} user@NodeID"
echo "then open http://localhost:${PORT} in your browser"

srun ${SRUN_ARGS} bash -lc "${APP_CMD}"


echo "END TIME: $(date)"
