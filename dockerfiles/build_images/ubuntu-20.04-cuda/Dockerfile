FROM rurabori/diploma:ubuntu-20.04

# add cuda remotes and keys.
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" >> /etc/apt/sources.list.d/cuda.list


RUN apt update
RUN apt install -y cuda-toolkit-11-3