from conans import ConanFile, CMake, tools, errors
import os


class CSPConan(ConanFile):
    name = "cmake_sample_project"
    author = "Boris Rúra boris.rura@avast.com"
    settings = "os", "compiler", "build_type", "arch"

    generators = "virtualenv", "cmake_paths", "cmake_find_package_multi"

    scm = {
        "type": "git",
        "subfolder": ".",
        "url": "auto",
        "revision": "auto"
    }

    requires = ['doctest/2.4.6', 'benchmark/1.5.5', 'fmt/8.0.1', 'tclap/1.2.4',
                'magic_enum/0.7.2', 'stx/1.0.1', 'scnlib/0.4', 'hdf5/1.12.0', 'spdlog/1.9.1']

    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.configure(source_folder=".")
        return cmake

    def configure(self):
        if self.settings.compiler == 'Visual Studio':
            del self.options.fPIC

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()
        cmake.test(args=['--verbose'])
