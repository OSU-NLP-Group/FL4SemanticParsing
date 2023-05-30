#!/usr/bin/env bash
WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

$(which mpirun) -np $PROCESS_NUM \
-hostfile mpi_host_file \
python3 torch_main.py --cf config/simulation/fedprox_8clients_mu_0.0001.yaml config/simulation/fedprox_8clients_mu_0.0001.json
