#!/bin/bash

############################################
# Add /usr/local/lib to the default library search path
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/local-lib.conf
ldconfig

############################################
# Set up / install GDB Eigen pretty printers
cp $WORKSPACE/scripts/.gdbinit /root
cp $WORKSPACE/scripts/.gdb_eigen /root

cd $WORKSPACE

exec bash
