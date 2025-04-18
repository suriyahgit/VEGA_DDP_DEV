export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=8  # 4 GPUs x 2 nodes

srun --ntasks=8 --nodes=2 --ntasks-per-node=4 \
     --gpus-per-task=1 --cpus-per-task=6 \
     --kill-on-bad-exit=1 \
     python -m torch.distributed.run \
       --nproc_per_node=4 \
       --nnodes=2 \
       --node_rank=$SLURM_NODEID \
       --master_addr=$MASTER_ADDR \
       --master_port=$MASTER_PORT \
       ddp_test.py

