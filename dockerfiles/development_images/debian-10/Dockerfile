FROM rurabori/diploma:debian-10

RUN apt install -y git gdb

# make the custom petsc installation visible.
ENV PKG_CONFIG_PATH=/usr/lib/petsc/lib/pkgconfig
ENV LD_LIBRARY_PATH=/usr/lib/petsc/lib
