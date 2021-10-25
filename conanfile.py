from conans import ConanFile, CMake, tools, errors
import os


class CSPConan(ConanFile):
    name = "cmake_sample_project"
    author = "Boris Rúra boris.rura@avast.com"
    settings = "os", "compiler", "build_type", "arch"

    generators = "virtualenv", "CMakeDeps"

    scm = {
        "type": "git",
        "subfolder": ".",
        "url": "auto",
        "revision": "auto"
    }

    requires = ['doctest/2.4.6', 'fmt/8.0.1', 'tclap/1.2.4', 'magic_enum/0.7.3',
                'stx/1.0.1', 'scnlib/0.4', 'hdf5/1.12.0', 'spdlog/1.9.2', 'structopt/0.1.2',
                'libcurl/7.78.0', 'libarchive/3.5.1'
                # , 'openmpi/4.1.0'
                ]

    def _configure_cmake(self):
        cmake= CMake(self)
        cmake.configure(source_folder=".")
        return cmake

    def configure(self):
        if self.settings.compiler == 'Visual Studio':
            del self.options.fPIC

        # can't have HDF5 C++ bindings in MPI enabled version.
        # self.options['hdf5'].parallel = True
        # self.options['hdf5'].enable_cxx = False
        # self.options['openmpi'].shared = True

    def build(self):
        cmake= self._configure_cmake()
        cmake.build()
        cmake.test(args=['--verbose'])
