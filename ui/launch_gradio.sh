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
MODEL_DIR="/capstor/store/cscs/swissai/a127/homes/mzhang/models/multimeditron/MultiMeditron-Llama-8B-Alignment-Generalist/checkpoint-314/"
BASE_PATH="/capstor/store/cscs/swissai/a127/homes/theoschiff"       # where our images live for FileSystemImageRegistry
PORT=49200
HOST="0.0.0.0"


echo "To access the app, after app initialisation, run:"
echo "ssh -J user@cluster -N -L ${PORT}:localhost:${PORT} user@NodeID"
echo "then open http://localhost:${PORT} in your browser"


python app.py \
  --model_checkpoint $MODEL_DIR \
  --base_path $BASE_PATH \
  --server_name ${HOST} \
  --server_port ${PORT}




