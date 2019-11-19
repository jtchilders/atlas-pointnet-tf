#!/bin/bash -l
#COBALT -q gpu_v100_smx2
#COBALT -n 1 
#COBALT -t 360
#COBALT -A datascience
#COBALT --jobname atlas-pointnet

source /gpfs/jlse-fs0/projects/datascience/parton/mlcuda/mconda/setup.sh
cd /gpfs/jlse-fs0/projects/datascience/parton/git/atlas-pointnet-tf
cp config.json $COBALT_JOBID.json
mpirun -n 8 python ./main.py -c config.json -l $COBALT_JOBID
echo $!
