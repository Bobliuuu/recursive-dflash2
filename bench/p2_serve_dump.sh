#!/bin/bash
# Start DSpine server (c=1 config) with the dump hook enabled. Usage: p2_serve_dump.sh /workspace/p2/dump.bin
mkdir -p $(dirname $1); rm -f $1
export RDF2_DUMP=$1
exec bash /workspace/recursive-dflash2/bench/serve.sh dspine
