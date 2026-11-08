#!/usr/bin/env bash
# Start a server for MODE, wait until ready, run the sweep, stop the server.  usage: run_method.sh NAME MODE [env assignments...]
# e.g. run_method.sh dspine16 dspine ; run_method.sh dspine8 dspine BLOCK=8
NAME=$1; MODE=$2; shift 2
W=${W:-/root/w}; cd $W
for kv in "$@"; do export "$kv"; done
pkill -f "[s]glang serve"; sleep 8
nohup ./serve.sh $MODE 30000 > $W/logs/srv_$NAME.log 2>&1 &
for i in $(seq 1 240); do grep -q "fired up" $W/logs/srv_$NAME.log && break; grep -q "Received sigquit" $W/logs/srv_$NAME.log && { echo "SERVER FAILED $NAME"; exit 1; }; sleep 5; done
$W/sweep.sh $NAME 30000
pkill -f "[s]glang serve"
