FROM debian:10

ARG cmake_version="3.21.1"

# prerequisites for adding new deb repos.
RUN apt update
RUN apt install -y gnupg ca-certificates software-properties-common curl libc6-dev

# install now so that apt doesn't try to update glibc.
RUN apt install -y curl libc6-dev
# development requirements
RUN apt install -y mpi-default-dev pkgconf petsc-dev python3 python3-pip

# add new repositories.
COPY dockerfiles/build_images/debian-10/additional_sources.list /etc/apt/sources.list.d/additional_sources.list

# update to get packages from added repos.
RUN apt update

# build requirements
RUN apt install -y ninja-build apt
# gcc-11
RUN apt install -y --no-install-recommends gcc-11 g++-11

RUN pip3 install conan

# install cmake.
RUN curl -L "https://github.com/Kitware/CMake/releases/download/v$cmake_version/cmake-$cmake_version-linux-x86_64.tar.gz" -o cmake.tar.gz && \
    tar xvfz cmake.tar.gz -C /usr/local && \
    rm cmake.tar.gz

# add cmake to path.
ENV PATH="/usr/local/cmake-$cmake_version-linux-x86_64/bin:${PATH}"
