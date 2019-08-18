include (ExternalProject)

set(cxxopts_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/cxxopts/include)
set(cxxopts_URL https://github.com/jarro2783/cxxopts.git)

set(cxxopts_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/cxxopts)

ExternalProject_Add(cxxopts
    PREFIX ${cxxopts}
    GIT_REPOSITORY ${cxxopts_URL}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cxxopts
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
