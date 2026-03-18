find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_PR gnuradio-PR)

FIND_PATH(
    GR_PR_INCLUDE_DIRS
    NAMES gnuradio/PR/api.h
    HINTS $ENV{PR_DIR}/include
        ${PC_PR_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_PR_LIBRARIES
    NAMES gnuradio-PR
    HINTS $ENV{PR_DIR}/lib
        ${PC_PR_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-PRTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_PR DEFAULT_MSG GR_PR_LIBRARIES GR_PR_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_PR_LIBRARIES GR_PR_INCLUDE_DIRS)
