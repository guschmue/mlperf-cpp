include (ExternalProject)

set(cxxopts_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/cxxopts/include)
set(cxxopts_URL https://github.com/jarro2783/cxxopts.git)

set(cxxopts_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/cxxopts)

ExternalProject_Add(cxxopts
    PREFIX ${cxxopts}
    GIT_REPOSITORY ${cxxopts_URL}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cxxopts
    BINARY_DIR ${cxxopts_BUILD_DIR}/cxxopts
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} ..
        ${cxxopts_EXTRA_OPT}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
)
