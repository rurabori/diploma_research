add_subdirectory(dim_cli)

# This is a bit of a dirty hack to force cmake --install --prefix to play nice
# with configure file and make this package actually relocatable.
install(
  CODE "configure_file(\"${CMAKE_HOME_DIRECTORY}/resources/modulefile.lua.in\"  \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_PROJECT_VERSION}.lua\")"
)
