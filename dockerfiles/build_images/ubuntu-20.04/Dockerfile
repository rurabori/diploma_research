FROM ubuntu:20.04


# prerequisites for adding new deb repos.
RUN apt update
RUN apt install -y gnupg ca-certificates software-properties-common

# install gcc 11 from PPA.
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && \
    apt install -y gcc-11 g++-11

# add cuda remotes and keys.
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" >> /etc/apt/sources.list.d/cuda.list

# add LLVM remotes and keys.
RUN apt-key adv --fetch-keys https://apt.llvm.org/llvm-snapshot.gpg.key && \
    echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-12 main" >> /etc/apt/sources.list.d/llvm.list

# update to get packages from added repos.
RUN apt update

RUN apt install -y clang-12 libomp-12-dev ninja-build curl python3 python3-pip cuda-toolkit-11-3 mpi-default-dev

RUN pip3 install conan

# install cmake.
RUN curl -L "https://github.com/Kitware/CMake/releases/download/v3.20.4/cmake-3.20.4-linux-x86_64.tar.gz" -o cmake.tar.gz && \
    tar xvfz cmake.tar.gz -C /usr/local && \
    rm cmake.tar.gz

# add cmake to path.
ENV PATH="/usr/local/cmake-3.20.4-linux-x86_64/bin:${PATH}"

# stdlibc++ has broken <ranges> header in this release (fixed in master). After the fixed is released we can remove this.
COPY dockerfiles/build_images/ubuntu-20.04/ranges.patch /patches/ranges.patch
RUN patch -R /usr/include/c++/11/ranges /patches/ranges.patch

