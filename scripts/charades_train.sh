#!/usr/bin/env bash

config=pool_charades_16x16_k5l8
version=w2gg
run_id=$(date +%Y%m%d_%H%M%S)

config_file=../configs/${config}.yaml
output_dir=../outputs/${config}/${version}/${run_id}

gpus=2
gpun=1
master_addr=127.0.0.9
master_port=29519

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${gpus}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using config_file=${config_file}"
echo "Using run_id=${run_id}"
echo "Using output_dir=${output_dir}"

python -c "import torch; print('torch visible cuda num:', torch.cuda.device_count()); print('current device:', torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

torchrun \
  --nproc_per_node=${gpun} \
  --master_addr ${master_addr} \
  --master_port ${master_port} \
  ../train_net.py \
  --config-file ${config_file} \
  OUTPUT_DIR ${output_dir}