#!/bin/bash
#SBATCH -t 05:30:00
#SBATCH --nodes=4               # node count. MUST BE ADDAPTED TO SPECIFIC PROBES AND MACHINES
#SBATCH --ntasks-per-node=4     # total number of tasks per node. MUST BE ADDAPTED TO SPECIFIC PROBES AND MACHINES
#SBATCH --cpus-per-task=32        # cpu-cores per task. MUST BE ADDAPTED TO SPECIFIC PROBES AND MACHINES
#SBATCH --mem=450G                # total memory per node. MUST BE ADDAPTED TO SPECIFIC PROBES AND MACHINES


module --force purge
module load Python/3.10
module load CUDA
source python_enviroment/bin/activate

export BATCH_SIZE=$1
export TRACE_FOLDER_NAME=$2

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT
export WORLD_SIZE=$SLURM_NPROCS
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export RANK=$SLURM_PROCID

echo Batch size: $BATCH_SIZE
echo Proceso: $SLURM_PROCID Local ID: $SLURM_LOCALID Nodo: $(hostname)


srun python BERT_sparsified_multinode.py
