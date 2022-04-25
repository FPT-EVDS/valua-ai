FROM continuumio/miniconda3 as builder

WORKDIR /app
ARG CONDA_DIR=/opt/conda

ENV PATH $CONDA_DIR/bin:$PATH

# Create the environment:
COPY flaskserver_py37.yml .
RUN conda env create -f flaskserver_py37.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "flaskserver_py37", "/bin/bash", "-c"]

RUN echo "source activate flaskserver_py37" > ~/.bashrc

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

RUN echo "Make sure cv2 is installed:"
RUN python -c "import cv2"

COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "flaskserver_py37", "python", "app.py"]