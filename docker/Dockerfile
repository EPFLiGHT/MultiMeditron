ARG NGC_VERSION="25.05-py3"

# To build documentation + cool utilies
RUN apt install graphviz, ffmpeg, tmux, sudo, git

FROM nvcr.io/nvidia/pytorch:$NGC_VERSION 
COPY . /workspace/MultiMeditron

WORKDIR /workspace/MultiMeditron
RUN pip install -e .

