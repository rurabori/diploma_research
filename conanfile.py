from conans import ConanFile, CMake, tools, errors
import os


class CSPConan(ConanFile):
    name = "cmake_sample_project"
    author = "Boris Rúra boris.rura@avast.com"
    settings = "os", "compiler", "build_type", "arch"

    generators = "virtualenv", "cmake_find_package_multi"

    scm = {"type": "git", "subfolder": ".", "url": "auto", "revision": "auto"}

    options = {
        "system_scientific_libs": [True, False],
        "enable_petsc_benchmark": [True, False],
    }

    default_options = {"system_scientific_libs": False, "enable_petsc_benchmark": False}

    requires = [
        "doctest/2.4.6",
        "fmt/8.0.1",
        "tclap/1.2.4",
        "magic_enum/0.7.3",
        "stx/1.0.1",
        "scnlib/0.4",
        "spdlog/1.9.2",
        "structopt/0.1.2",
        "libcurl/7.78.0",
        "libarchive/3.5.1",
        "yaml-cpp/0.7.0",
        "benchmark/1.6.0",
        "nlohmann_json/3.10.4",
        "opencv/4.5.3",
    ]

    def requirements(self):
        if bool(self.options.system_scientific_libs):
            return

        if self.options.enable_petsc_benchmark:
            self.requires("petsc/3.16.0@rurabori/stable")

        self.requires("hdf5/1.12.0@rurabori/stable")

    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.configure(source_folder=".")
        return cmake

    def configure(self):
        self.options["opencv"].with_gtk = False
        self.options["opencv"].with_eigen = False
        self.options["opencv"].with_webp = False
        self.options["opencv"].with_quirc = False
        self.options["opencv"].with_tiff = False
        self.options["opencv"].with_ffmpeg = False

        if self.settings.compiler == "Visual Studio":
            del self.options.fPIC

        # can't have HDF5 C++ bindings in MPI enabled version.
        # self.options['hdf5'].parallel = True
        # self.options['hdf5'].enable_cxx = False
        # self.options['openmpi'].shared = True

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()
        cmake.test(args=["--verbose"])
