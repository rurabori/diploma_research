FROM debian:10

ARG cmake_version="3.21.1"


RUN apt update
# install now so that apt doesn't try to update glibc.
RUN apt install -y curl libc6-dev

# add additional repositories and keys.
RUN apt install -y gnupg ca-certificates software-properties-common
RUN apt-key adv --fetch-keys https://apt.llvm.org/llvm-snapshot.gpg.key
COPY dockerfiles/build_images/debian-10/additional_sources.list /etc/apt/sources.list.d/additional_sources.list

# now we should have new packages.
RUN apt update

# development requirements
RUN apt install -y mpi-default-dev pkgconf petsc-dev python3 python3-pip
# build requirements
RUN apt install -y ninja-build
RUN apt install -y --no-install-recommends clang-12 libomp-12-dev libc++-12-dev libc++abi-12-dev

RUN pip3 install conan

# install cmake.
RUN curl -L "https://github.com/Kitware/CMake/releases/download/v$cmake_version/cmake-$cmake_version-linux-x86_64.tar.gz" -o cmake.tar.gz && \
    tar xvfz cmake.tar.gz -C /usr/local && \
    rm cmake.tar.gz

# add cmake to path.
ENV PATH="/usr/local/cmake-$cmake_version-linux-x86_64/bin:${PATH}"
