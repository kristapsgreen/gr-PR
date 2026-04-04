#!/usr/bin/sh
export VOLK_GENERIC=1
export GR_DONT_LOAD_PREFS=1
export srcdir="/home/kristaps/BAK/gr-PR/python/PR"
export GR_CONF_CONTROLPORT_ON=False
export PATH="/home/kristaps/BAK/gr-PR/build/python/PR":"$PATH"
export LD_LIBRARY_PATH="":$LD_LIBRARY_PATH
export PYTHONPATH=/home/kristaps/BAK/gr-PR/build/test_modules:$PYTHONPATH
/usr/bin/python3 /home/kristaps/BAK/gr-PR/python/PR/qa_caf.py 
