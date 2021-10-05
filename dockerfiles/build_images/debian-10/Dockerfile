FROM debian:10

ARG cmake_version="3.21.1"

RUN apt update
# install now so that apt doesn't try to update glibc.
RUN apt install -y curl libc6-dev

# add additional repositories and keys.
RUN apt install -y gnupg ca-certificates software-properties-common
RUN apt-key adv --fetch-keys https://apt.llvm.org/llvm-snapshot.gpg.key
COPY dockerfiles/build_images/debian-10/additional_sources.list /etc/apt/sources.list.d/additional_sources.list

COPY dockerfiles/build_images/debian-10/petsc-3.15.tar.gz /usr/lib/petsc-3.15.tar.gz
RUN cd /usr/lib && tar xvfz petsc-3.15.tar.gz && rm petsc-3.15.tar.gz

# now we should have new packages.
RUN apt update

# development requirements
RUN apt install -y mpi-default-dev pkgconf python3 python3-pip
# build requirements
RUN apt install -y ninja-build
RUN apt install -y clang-13 libomp-13-dev libc++-13-dev libc++abi-13-dev libunwind-dev

RUN pip3 install conan

# install cmake.
RUN curl -L "https://github.com/Kitware/CMake/releases/download/v$cmake_version/cmake-$cmake_version-linux-x86_64.tar.gz" -o cmake.tar.gz && \
    tar xvfz cmake.tar.gz -C /usr/local && \
    rm cmake.tar.gz

# add cmake to path.
ENV PATH="/usr/local/cmake-$cmake_version-linux-x86_64/bin:${PATH}"
