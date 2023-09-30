num_nodes=1
num_gpus=`nvidia-smi --list-gpus | wc -l`
master_addr=`hostname`
master_port=`shuf -i 2000-65000 -n 1`
echo "master address is ${master_addr}:${master_port}"

torchrun --nproc_per_node=${num_gpus} \
        --nnodes=${num_nodes} \
        --node_rank=0  \
        --master_addr=${master_addr} \
        --master_port=${master_port} \
        ./dino_training/train.py --config $@