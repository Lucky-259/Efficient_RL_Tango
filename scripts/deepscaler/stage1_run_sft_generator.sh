#!/bin/bash
set -x

train_files=./data/eurus2_sft_math/deepseek_r1_distill_llama70b_sft_data_generation_train.parquet
val_files=./data/eurus2_sft_math/deepseek_r1_distill_llama70b_sft_data_generation_test.parquet
model_path=${your_base_models_path}/DeepScaleR-1.5B-Preview
project_name='RL-Tango'
experiment_name='sft-generator-deepscaler-1.5b-preview'
save_path=./checkpoints/$project_name/$experiment_name

# number of nodes
NNODES=${NNODES:-1}
# rank of current node
NODE_RANK=${NODE_RANK:-0}
# ip address of master node
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
# port of master node
MASTER_PORT=${MASTER_PORT:-29500}
# number of processes per node
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nnodes)
      NNODES="$2"
      shift 2
      ;;
    --node_rank)
      NODE_RANK="$2"
      shift 2
      ;;
    --master_addr)
      MASTER_ADDR="$2"
      shift 2
      ;;
    --master_port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --nproc_per_node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NPROC_PER_NODE \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_batch_size=64 \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.prompt_key=prompt \
    data.response_key=responses \
    data.max_length=4096 \
    data.micro_batch_size_per_gpu=2 \
    data.apply_chat_template=False \
    optim.lr=5e-6 \
    model.partial_pretrain=$model_path \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=2 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null "$@"