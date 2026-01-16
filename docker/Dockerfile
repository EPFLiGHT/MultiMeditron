ARG NGC_VERSION="25.05-py3"

FROM nvcr.io/nvidia/pytorch:$NGC_VERSION

# To build documentation + cool utilies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        graphviz \
        ffmpeg \
        tmux \
        sudo \
        git 

COPY . /workspace/MultiMeditron
WORKDIR /workspace/MultiMeditron
RUN pip install -e ".[flash-attn]"
RUN pip install git+https://github.com/EPFLiGHT/lmms-eval.git
