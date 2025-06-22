# -----------------------------------------------------------------------------
# Base Image - Universal Base Image (UBI) 9 with Python 3.9
# -----------------------------------------------------------------------------
FROM registry.access.redhat.com/ubi9/python-39:9.6-1749743801

    USER root
    # -----------------------------------------------------------------------------
    # CUDA Configuration
    # -----------------------------------------------------------------------------
    # add the NVIDIA CUDA repository (RHEL9)
    RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && \
        dnf clean all

    # Install necessary components for CUDA 12.8 to enable GPU acceleration:
    # - cuda-cudart-devel: Development libraries for CUDA runtime
    # - cuda-nvcc: NVIDIA CUDA compiler
    # - cuda-compiler: Meta package for compiler tools
    # - cuda-libraries-devel: Libraries for CUDA-based development
    RUN dnf install -y \
            cuda-cudart-devel-12-8 \
            cuda-nvcc-12-8 \
            cuda-compiler-12-8 \
            cuda-libraries-devel-12-8

    # Configure environment variables to include CUDA binaries and libraries
    ENV PATH="/usr/local/cuda-12.8/bin:${PATH}"
    ENV LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64"

    # -----------------------------------------------------------------------------
    # Build Tools Installation
    # -----------------------------------------------------------------------------
    RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.2/cmake-3.29.2-linux-x86_64.sh && \
        sh cmake-3.29.2-linux-x86_64.sh --skip-license --prefix=/usr/local && \
        rm cmake-3.29.2-linux-x86_64.sh

    # -----------------------------------------------------------------------------
    # Python Dependencies
    # -----------------------------------------------------------------------------
    RUN pip3 install --upgrade pip && \
        pip3 install \
            torch==2.7.1 \
            torchvision==0.22.1 \
            --index-url https://download.pytorch.org/whl/cu128

    RUN pip3 install wheel ninja==1.11.1.4

    # -----------------------------------------------------------------------------
    # Application Configuration
    # -----------------------------------------------------------------------------
    ARG env_det_num_threads=6
    ARG env_det_verbose=1
    ARG OMP_NUM_THREADS=1

    ENV det_data=/opt/data \
        det_models=/opt/models \
        det_num_threads=$env_det_num_threads \
        det_verbose=$env_det_verbose \
        OMP_NUM_THREADS=$OMP_NUM_THREADS

    # -----------------------------------------------------------------------------
    # Application Installation
    # -----------------------------------------------------------------------------

    # FIXME: Docker doesn't provide a way to expose and mount gpu devices during
    # build stage. set TORCH_CUDA_ARCH_LIST to PTX to avoid errors during  build
    # stage. The user should set TORCH_CUDA_ARCH_LIST  to  theappropriate  value
    # for their GPU architecture when building the image. This can  be  done  by
    # passing       I.E        --build-arg        TORCH_CUDA_ARCH_LIST="8.0;8.6"
    # https://en.wikipedia.org/wiki/CUDA#GPUs_supported

    ARG TORCH_CUDA_ARCH_LIST= "PTX"
    ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST

    RUN mkdir -p /opt/code/nndet
    WORKDIR /opt/code/nndet
    COPY . .

    # Install the application with the following configurations:
    # - --no-build-isolation: Ensures the  build  utilizes  the  pre-installed
    #   CUDA environment and dependencies
    # - FORCE_CUDA=1: Explicitly enables CUDA support during the build process
    RUN FORCE_CUDA=1 pip install --no-build-isolation -e .