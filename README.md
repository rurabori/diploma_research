# DIM: distributed matrix multiplication

[Thesis](https://rurabori.jfrog.io/ui/repos/tree/General/default-generic-local%2Fdim%2Fthesis%2Flatest.pdf), right-click on latest to download the artifact.

## Dependencies 

This project uses `conan` as its package manager, which can be obtained by running:
```bash
pip install conan
```

After obtaining conan it should be invoked by cmake during configuration so no additional steps 
are necessary. If dim isn't being built on RCI cluster, and PETSc is required, my artifactory 
needs to be added to conan remotes. This can be achieved by running:

```bash
# the remote requires revisions to 
# be enabled in local conan installation
conan config set general.revisions_enabled=1
# this hosts the PETSc package recipe as well
# as some prebuilt libraries.
conan remote add rurabori-conan \
    https://rurabori.jfrog.io/artifactory/api/conan/rurabori-conan
```

## Configuration

The project provides only two configuration options.

+ `enable_petsc_benchmark` which is disabled by default. If it is enabled,
the PETSc implementation of the distributed conjugate gradient will be built, and the project
will require PETSc as a dependency. 
+ `system_scientific_libs` which tells the benchmarks to use system scientific libs. This is useful 
for building on RCI cluster.

Options can be enabled/disabled by passing `-D<option>"ON/OFF"` to `cmake` during configuration phase.
Example of configuration command line on RCI cluster:

```bash
cmake . -B build \
        -Dsystem_scientific_libs="ON" \
        -Denable_petsc_benchmark="ON" \
        -DENABLE_TESTING="OFF" \
        -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release
```

## Build 

To build just run:

```bash
cmake --build <build_folder> --parallel <njobs>
```

## Installation

The project can be installed by running:

```bash
cmake --install --prefix <install dir>
```

This will install the `dim` CLI to the specified directory and create a modulefile for SLURM 
system used on RCI.