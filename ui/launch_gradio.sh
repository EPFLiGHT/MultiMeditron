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


SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"         # -> .../meditron/MultiMeditron
APP_REL="ui/app.py"                                # app lives in ui/
# MODEL_DIR="/capstor/store/cscs/swissai/a127/homes/theoschiff/models/MultiMeditron-8B-Clip/checkpoint-813"
# BASE_PATH="/capstor/store/cscs/swissai/a127/homes/theoschiff"       # where our images live for FileSystemImageRegistry
MODEL_DIR="/capstor/store/cscs/swissai/a127/homes/mzhang/models/multimeditron/MultiMeditron-Llama-8B-Alignment-Generalist/checkpoint-314/"
PORT=49200
HOST="0.0.0.0"

echo "Running app from ${REPO_DIR}/${APP_REL}"

python app.py \
  --model_checkpoint $MODEL_DIR \
  --server_name ${HOST} \
  --server_port ${PORT}




