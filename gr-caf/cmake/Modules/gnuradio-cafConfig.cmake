find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_CAF gnuradio-caf)

FIND_PATH(
    GR_CAF_INCLUDE_DIRS
    NAMES gnuradio/caf/api.h
    HINTS $ENV{CAF_DIR}/include
        ${PC_CAF_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_CAF_LIBRARIES
    NAMES gnuradio-caf
    HINTS $ENV{CAF_DIR}/lib
        ${PC_CAF_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-cafTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_CAF DEFAULT_MSG GR_CAF_LIBRARIES GR_CAF_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_CAF_LIBRARIES GR_CAF_INCLUDE_DIRS)
