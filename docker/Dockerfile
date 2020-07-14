FROM nvidia/cuda:10.1-devel-ubuntu18.04

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.1
ENV NCCL_VERSION=2.4.8-1+cuda10.1
ARG PYTHON_VERSION=3.7
ARG WITH_TORCHVISION=1

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse" >> /etc/apt/sources.list

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        openssh-server \
        openssh-client \
        screen && \
        rm -rf /var/lib/apt/lists/*

# install python & pytorch
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml \
                                     scipy matplotlib jupyter jupyterlab \
                                     ipython mkl mkl-include ninja cython typing \
                                     scikit-learn pandas opencv bcolz&& \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_PYTORCH=1 \
         pip install --no-cache-dir horovod && \
    ldconfig

## ssh config
ARG RSA_PUB="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCx11vMvwr+9nIUqXewbyDFmKtzD6eLGugUgVi+0v0zvClhTTOUWNG8JuQz5qbeetwUY6zErwH1JJhZK6PW1SYKpnodIXKStqAUt+QHl4q3g3b+rKSBhL8gj+6fNRi6MToDGVYlmb+PPXe71p6g1zm2m6qAw2Ey6CQ1r84EFJJNQHoPQbSiPatE54wPl9a5K/eEQIuvZbKH0HkMQmn65Jw9rYO6zFbyxCfkSBUsNpx/U19vyWR0XpQSUC27W35DFm2mTrkSfU7dbBgekwBGj+MylSzYXP+iAF/GKw235BeqQcOdzJsgVGvO30Gt2kpQdPnY3WAcFL0YSYzhH6I4aLXUIBHKLD0fPtt91I0BwfG832GMnAxXCFZzZHZFT72mXsHPSEkqWNsu7+jwKSIoyt75LpWcDjHWp4dnkU1BFgkkg1aN70X2Y+YWsUcFsHY36fbvxUb7t9HSRtN0dkB8A/dLsJGarE2BJQhY60rnIMsk6e7CB3QvWh9ID7gHmyWRCY94qgpwZ37bDyVKQ+tS1/uQm2mZt7F5yTK1dSJCvGq+mJ5/YaQ+359FTBWLUP7bufUxUYwJ04a7Alrvfz8tGs+sY9ykfyKBFlXjPLrMRbBBdUG+AXBpIgMZQZShXE+bgidPmX4Qg8EVan+JexBxx3ylQNAaGWxFwPpsBF06gToW9Q== renjie_xie@seu.edu.cn"

RUN mkdir -p /var/run/sshd && mkdir -p /root/.ssh && \
    sed -ri "s/session required pam_loginuid.so/#session required pam_loginuid.so/g" /etc/pam.d/sshd && \
    cat $RSA_PUB > /root/.ssh/authorized_keys

## import rsa_pub from file
# RUN mkdir -p /var/run/sshd && mkdir -p /root/.ssh && \
#     sed -ri "s/session required pam_loginuid.so/#session required pam_loginuid.so/g" /etc/pam.d/sshd
# COPY ./authorized_keys /root/.ssh/authorized_keys

RUN echo "#! /bin/bash" > /run.sh && echo "/etc/init.d/ssh start -D" >> /run.sh && \
    chmod 755 /run.sh && mkdir -p /workspace

EXPOSE 22

WORKDIR /workspace
RUN chmod -R a+w .

CMD ["/run.sh"]
