ARG UBUNTU_VERSION=18.04

ARG ARCH=
ARG CUDA=10.1
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.4.38-1
ARG CUDNN_MAJOR_VERSION=7
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=6.0.1-1
ARG LIBNVINFER_MAJOR_VERSION=6

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        # There appears to be a regression in libcublas10=10.2.2.89-1 which
        # prevents cublas from initializing in TF. See
        # https://github.com/tensorflow/tensorflow/issues/9489#issuecomment-562394257
        libcublas10=10.2.1.243-1 \ 
        cuda-nvrtc-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

# Install TensorRT if not building for PowerPC
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3.8-dev \
    libexpat1-dev \
    libicu-dev \
    libigraph0-dev

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 ./get-pip.py

RUN python3.8 -m pip install pip && \
    pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3.8) /usr/local/bin/python

ARG TF_PACKAGE=tensorflow-gpu
ARG TF_PACKAGE_VERSION
RUN pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

# COPY bashrc /etc/bash.bashrc
# RUN chmod a+rwx /etc/bash.bashrc 
# COPY bashrc ~.bashrc
# RUN chmod a+rwx ~.bashrc 

ARG AI_USER="ai"
ARG AI_UID="1001"

USER root

# Install all OS dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    run-one \
    systemd \
    ssh \
    acl \
    nano \
    libsndfile-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen
    
# Configure environment
ENV HOME=/home/$AI_USER

# Enable prompt color in the skeleton .bashrc before creating the default AI_USER
# RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc

RUN useradd -m -s /bin/bash -N -u $AI_UID $AI_USER 
RUN usermod -aG sudo $AI_USER

WORKDIR $HOME


# Install and setup ssh access
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo $AI_USER:aeroplanelikestodrink | chpasswd

CMD ["/usr/sbin/sshd", "-D"]

# Install Python libs
RUN pip install --no-cache-dir \
    spacy~=2.2.4 \
    pandas~=1.0.1 \
    httplib2==0.11.3 \
    urllib3==1.24.1 \
    numpy~=1.18.2 \
    scikit-learn~=0.22.2.post1 \
    Scrapy~=2.0.0 \
    textblob~=0.15.3 \
    numba~=0.48.0 \
    bayesian-optimization \tensorflow
    numba~=0.48.0 \
    pandas~=1.0.4 \
    speechpy \
    soundfile \
    keras-tuner \
    gensim~=3.8.3 \
    scipy~=1.4.1 \
    scikit-learn \
    tensorflow-addons==0.10.0 \
    tensorflow-probability==0.10.1

    
# Install Python libraries that were not yet installed 
# New libs should be added to Dockerfile -- https://jpetazzo.github.io/2013/12/01/docker-python-pip-requirements/
# ADD requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt 


