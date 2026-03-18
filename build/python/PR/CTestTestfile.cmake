# CMake generated Testfile for 
# Source directory: /home/kristaps/BAK/gr-PR/python/PR
# Build directory: /home/kristaps/BAK/gr-PR/build/python/PR
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(qa_caf "/usr/bin/sh" "qa_caf_test.sh")
set_tests_properties(qa_caf PROPERTIES  _BACKTRACE_TRIPLES "/usr/lib/x86_64-linux-gnu/cmake/gnuradio/GrTest.cmake;116;add_test;/home/kristaps/BAK/gr-PR/python/PR/CMakeLists.txt;43;GR_ADD_TEST;/home/kristaps/BAK/gr-PR/python/PR/CMakeLists.txt;0;")
add_test(qa_rd_plot "/usr/bin/sh" "qa_rd_plot_test.sh")
set_tests_properties(qa_rd_plot PROPERTIES  _BACKTRACE_TRIPLES "/usr/lib/x86_64-linux-gnu/cmake/gnuradio/GrTest.cmake;116;add_test;/home/kristaps/BAK/gr-PR/python/PR/CMakeLists.txt;44;GR_ADD_TEST;/home/kristaps/BAK/gr-PR/python/PR/CMakeLists.txt;0;")
subdirs("bindings")
