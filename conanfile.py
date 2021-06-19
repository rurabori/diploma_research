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

    requires = ['doctest/2.4.0', 'benchmark/1.5.2',
                'fmt/7.1.3', 'tclap/1.2.3', 'magic_enum/0.7.2']

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
