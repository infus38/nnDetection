# Contains pytorch, torchvision, cuda, cudnn
FROM nvcr.io/nvidia/pytorch:21.11-py3

ARG env_det_num_threads=6
ARG env_det_verbose=1

# Setup environment variables
ENV det_data=/opt/data det_models=/opt/models det_num_threads=$env_det_num_threads det_verbose=$env_det_verbose OMP_NUM_THREADS=1

# Install some tools
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y \
 git \
 cmake \
 make \
 wget \
 gnupg \
 build-essential \
 software-properties-common \
 gdb \
 ninja-build

# updating requests and urllib3 fixed compatibility with my docker version
RUN pip install numpy \
  && pip install --upgrade requests \
  && pip install --upgrade urllib3

# Install own code
COPY ./requirements.txt .
RUN mkdir ${det_data} \
  && mkdir ${det_models} \
  && mkdir -p /opt/code/nndet \
  && pip install -r requirements.txt  \
  && pip install hydra-core --upgrade --pre \
  && pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git

WORKDIR /opt/code/nndet
COPY . .
RUN FORCE_CUDA=1 pip install -v -e .
