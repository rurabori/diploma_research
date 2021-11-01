FROM ubuntu:20.04

ARG cmake_version="3.21.1"

# prerequisites for adding new deb repos.
RUN apt update
RUN apt install -y gnupg ca-certificates software-properties-common

# install gcc 11 from PPA.
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && \
    apt install -y gcc-11 g++-11

# add LLVM remotes and keys.
RUN apt-key adv --fetch-keys https://apt.llvm.org/llvm-snapshot.gpg.key && \
    echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-13 main" >> /etc/apt/sources.list.d/llvm.list

# update to get packages from added repos.
RUN apt update

RUN apt install -y clang-13 clang-tidy-13 libunwind-13-dev libomp-13-dev ninja-build curl python3 python3-pip mpi-default-dev pkgconf petsc-dev

RUN pip3 install conan

# install cmake.
RUN curl -L "https://github.com/Kitware/CMake/releases/download/v$cmake_version/cmake-$cmake_version-linux-x86_64.tar.gz" -o cmake.tar.gz && \
    tar xvfz cmake.tar.gz -C /usr/local && \
    rm cmake.tar.gz

# add cmake to path.
ENV PATH="/usr/local/cmake-$cmake_version-linux-x86_64/bin:${PATH}"

# stdlibc++ has broken <ranges> header in this release (fixed in master). After the fixed is released we can remove this.
COPY dockerfiles/build_images/ubuntu-20.04/ranges.patch /patches/ranges.patch
RUN patch -R /usr/include/c++/11/ranges /patches/ranges.patch

RUN conan remote add rurabori-conan https://rurabori.jfrog.io/artifactory/api/conan/rurabori-conan
