FROM continuumio/miniconda3 as builder

WORKDIR /app
ARG CONDA_DIR=/opt/conda

ENV PATH $CONDA_DIR/bin:$PATH

# Create the environment:
COPY flaskserver_py37.yml .
RUN conda env create -f flaskserver_py37.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "flaskserver_py37", "/bin/bash", "-c"]

#STAGE 2
#FROM nvidia/cuda:10.1-cudnn7-runtime
#
#WORKDIR /app

#RUN apt update && \
#    apt install --no-install-recommends -y build-essential software-properties-common && \
#    add-apt-repository -y ppa:deadsnakes/ppa && \
#    apt clean && rm -rf /var/lib/apt/lists/*
#COPY --from=builder /opt/conda/. $CONDA_DIR

RUN echo "source activate flaskserver_py37" > ~/.bashrc
#ENV PATH /opt/conda/envs/env/bin:$PATH

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

RUN echo "Make sure cv2 is installed:"
RUN python -c "import cv2"

#RUN echo "Make sure torch.cuda is installed:"
#RUN python -c "import torch"
#RUN python -c "torch.cuda.is_available()"

# The code to run when container is started:
COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "flaskserver_py37", "python", "app.py"]