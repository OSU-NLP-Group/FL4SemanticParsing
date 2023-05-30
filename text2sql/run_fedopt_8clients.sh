#!/usr/bin/env bash
WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

$(which mpirun) -np $PROCESS_NUM \
-hostfile mpi_host_file \
python3 torch_main.py --cf config/simulation/fedopt_8clients.yaml config/simulation/fedopt_8clients.json
