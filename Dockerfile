FROM nvcr.io/nvidia/pytorch:24.10-py3

RUN apt-get update && apt-get install -y openssh-server && apt-get install -y pssh && apt-get install -y tmux && service ssh start
EXPOSE 22
