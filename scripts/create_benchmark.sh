name="${1}"

appdir="benchmarks/$name"
mkdir -p $appdir

echo -e "brr_add_executable($name)\ninstall(TARGETS $name RUNTIME DESTINATION \${CMAKE_INSTALL_BINDIR})\n" >"$appdir/CMakeLists.txt"
echo -e "int main(int argc, const char* argv[]) { return 0; }\n" >"$appdir/main.cpp"
echo -e "add_subdirectory($name)" >>"benchmarks/CMakeLists.txt"
