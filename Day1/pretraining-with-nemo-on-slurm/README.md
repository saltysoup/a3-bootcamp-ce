# Instructions for pre-training llama2 7b using NeMo on Slurm

# Use exactly 2 x A3 mega VMs per CE only, in asia-northeast1-b zone

## ***Objective***
### *By following this lab, you will be able to deploy a A3 plus Slurm cluster to run a llama2 pre-training job using Nvidia NeMo framework*

## ***Provisioning a cluster***

## Step 1: Use the public deployment guide, with minor alterations

**Note: It's recommended to not use cloudshell for the following, as session timeout can impact the long terraform deployment time**

```
project_id: injae-sandbox-340804
region: asia-northeast1
zone: asia-northeast1-b
```

1. Install the Cloud HPC toolkit in your local environment, including its dependencies [using this link.](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster)

2. [Follow the public deployment guide](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster) with the following changes:

3. A reservation has already been created for you (you don't need to create a new one). Use the `a3mega-bootcamp` reservation when updating the [cluster deployment file](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster#update-deployment) later

4. Make the following changes in the [base deployment file](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster#update-filestore-deployment)

- `deployment_name: yourUsername-base`
- `network_name_system: yourUsername-net`
- `subnetwork_name_system: yourUsername-subnet`

5. Before [deploying the VPC and Filestore](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster#setup-filestore), Make changes in `slurm-a3mega-base.yaml`. Change the values to `filestore_tier: BASIC_SSD` and `size_gb: 2560` in line 53-54.
- **Note the private ip address of your Filestore instance in the deployment output. You will need this later!**

6. For [Update the cluster deployment file](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster#update-deployment) step, make the following changes in `deployment-image-cluster.yaml`.

- `deployment_name: yourLdap-cluster`
- `network_name_system: yourLdap-net`
- `subnetwork_name_system: yourLdap-subnet`
- `slurm_cluster_name: yourLdap`
- `a3mega_reservation_name: a3mega-bootcamp`
- `a3mega_maintenance_interval: ""` - As we are not using a gSC reservation
- `a3mega_cluster_size: 2`
- `server_ip_homefs: yourFilestoreIP` - From the previous deployment output


7. To save ~20 min, an custom OS image has already been built ahead of time, called `final_image_family: slurm-a3mega`. It's recommended to skip the [Build the custom OS image](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster#build-image) step

8. Make the following changes in the `slurm-a3mega-cluster.yaml`

- `deployment_name: yourLdap-cluster`

9. Proceed to the [Provision the Slurm cluster](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster#provision-cluster) step. This will take approx 10 min to complete.


## ***Validate network performance on the cluster with a NCCL test***

## Step 1: Use the public documented guide

1. To perform a basic performance validation of the cluster, we'll be running a NCCL test across 2 x A3 Mega nodes. This will verify that multi-NIC performance using Fastrak is functional, which is critical for optimising distributed training.

2. SSH into the newly created Slurm login node using the provided lab user credentials.
- `This should be yourLdap@ikwak.altostrat.com`

3. Follow the guide on https://cloud.google.com/hpc-toolkit/docs/machine-learning/a3-mega-enable-gpudirect-tcpxo

**Note: Note that first job will take ~10 min to run as the A3 Mega nodes need to download the NCCL and RxDM container for Fastrak to work. You can verify this by SSH'ing into a A3 mega VM and using `docker images`.**

4. You should get a result of ~180GB/s for 8589934592 message size (8GB) after running the NCCL all_reduce network test. This indicates that 1440 Gbps out of the maximum 1600 Gbps VM networking is being used. With topology awareness and gSC deployment, the results will be ~10% higher.

Example (see values under `busbw` column)
```
 0: #                                                              out-of-place                       in-place          
 0: #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
 0: #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
 0:      8388608        131072     float    none      -1    350.9   23.91   22.41    N/A    345.0   24.31   22.79    N/A
 0:     16777216        262144     float    none      -1    394.7   42.50   39.85    N/A    392.6   42.73   40.06    N/A
 0:     33554432        524288     float    none      -1    444.9   75.43   70.71    N/A    441.8   75.94   71.20    N/A
 0:     67108864       1048576     float    none      -1    705.8   95.08   89.14    N/A    709.0   94.65   88.74    N/A
 0:    134217728       2097152     float    none      -1   1101.8  121.82  114.21    N/A   1101.9  121.81  114.20    N/A
 0:    268435456       4194304     float    none      -1   2007.2  133.73  125.38    N/A   2002.6  134.04  125.66    N/A
 0:    536870912       8388608     float    none      -1   3490.1  153.83  144.21    N/A   3383.0  158.70  148.78    N/A
 0:   1073741824      16777216     float    none      -1   6052.2  177.41  166.32    N/A   6040.9  177.74  166.64    N/A
 0:   2147483648      33554432     float    none      -1    11520  186.41  174.76    N/A    11363  188.98  177.17    N/A
 0:   4294967296      67108864     float    none      -1    21720  197.74  185.38    N/A    21642  198.46  186.05    N/A
 0:   8589934592     134217728     float    none      -1    42954  199.98  187.48    N/A    43188  198.90  186.47    N/A
 0: # Out of bounds values : 0 OK
 0: # Avg bus bandwidth    : 120.346 
```

## ***Run llama2 training job on NeMo***

## Step 1: Set up NeMo in your Slurm cluster

1. SSH into your newly created Slurm login node, clone the HPC toolkit repo

```
cd ~
git clone https://github.com/GoogleCloudPlatform/hpc-toolkit.git && cd hpc-toolkit/examples/machine-learning/a3-megagpu-8g/nemo-framework
```

2. Update `Dockerfile` to use the latest NeMo container. This will pull the latest NeMo container from Nvidia and set environment variables required for Fastrak.

```
ARG NEMOFW_VERSION=24.05.llama3.1
FROM nvcr.io/nvidia/nemo:${NEMOFW_VERSION}

ENV NCCL_FASTRAK_CTRL_DEV=enp0s12
ENV NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0
ENV NCCL_SOCKET_IFNAME=enp0s12
ENV NCCL_CROSS_NIC=0
ENV NCCL_ALGO=Ring,Tree
ENV NCCL_PROTO=Simple
ENV NCCL_MIN_NCHANNELS=4
ENV NCCL_DYNAMIC_CHUNK_SIZE=524288
ENV NCCL_P2P_NET_CHUNKSIZE=524288
ENV NCCL_P2P_PCI_CHUNKSIZE=524288
ENV NCCL_P2P_NVL_CHUNKSIZE=1048576
ENV NCCL_FASTRAK_NUM_FLOWS=2
ENV NCCL_FASTRAK_USE_SNAP=1
ENV NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
ENV NCCL_BUFFSIZE=8388608
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_NET_GDR_LEVEL=PIX
ENV NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
ENV NCCL_FASTRAK_USE_LLCM=1
ENV NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices
ENV NCCL_TUNER_CONFIG_PATH=/var/lib/tcpxo/lib64/a3plus_tuner_config.textproto

RUN echo "/var/lib/tcpxo/lib64" >> /etc/ld.so.conf.d/tcpxo.conf && ldconfig
ENV LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH
```

3. Update `setup_nemo.sh` to use the latest NeMo container. This will submit a Slurm job to a A3 Mega VM for building a new [squash file](https://github.com/NVIDIA/enroot/blob/master/doc/image-format.md) using the container image to allow [Enroot](https://github.com/NVIDIA/enroot) and [Pyxis](https://github.com/NVIDIA/pyxis) to run container workloads on Slurm.

```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a3mega
#SBATCH --exclusive

: "${NEMOFW_VERSION:=24.05.llama3.1}"

srun docker build --build-arg="NEMOFW_VERSION=${NEMOFW_VERSION}" -t nemofw:tcpxo-"${NEMOFW_VERSION}" .
srun rm -f nemofw+tcpxo-"${NEMOFW_VERSION}".sqsh
srun enroot import dockerd://nemofw:tcpxo-"${NEMOFW_VERSION}"

srun \
        --container-mounts="${PWD}":/workspace/mount_dir,/var/tmp:/var/tmp \
        --container-image=./nemofw+tcpxo-"${NEMOFW_VERSION}".sqsh \
        bash -c "cp -r /opt/NeMo-Framework-Launcher/requirements.txt /opt/NeMo-Framework-Launcher/launcher_scripts /opt/NeMo-Framework-Launcher/auto_configurator /workspace/mount_dir/"
```

4. Set up NeMo container by submitting a slurm job. This will build a local sqsh file in your filestore home directory, and make a local copy of the scripts and requirements.txt file.
```
sbatch setup_nemo.sh
```

5. Monitor the job using the following command. This should take approx. 15 min to complete and will show the Status as "R" or Running. The job will automatically disappeart from the queue once it is completed. 
```
squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
                 3    a3mega setup_ne ext_ikwa  R       1:03      1 ikwak-a3meganodese
```

6. If successful, you should now see `requirements.txt`, `launcher_scripts` and `auto_configurator` in your current directory

7. Install NeMo Framework required packages in a new python virtual environment from the login node
```
python3 -m venv nemo_env
source nemo_env/bin/activate
pip install -r requirements.txt # Copied from the NeMo Framework Container earlier
# This is needed to use 23.11 and python3.11, which is what is present on Debian 12.
# nvitop is a useful tool to real time monitor GPU utilisation and processes for debugging/optimisation 
pip install -U hydra-core nvitop
```

## Step 2: Run and optimise llama2 training job in your Slurm cluster

In this step, we'll be running a llama2 training job. Refer to yaml examples in `launcher_scripts/conf/training/llama` for pre-configured NeMo configurations for various llama models.

1. Create a new NeMo training config for llama2. This will include the latest bucket merge optimisation from the [NeMo Github repo](https://github.com/NVIDIA/NeMo/tree/main), which enables better communication overlap for faster training.

```
# Create a new training config from the launcher_scripts directory

touch conf/training/llama/llama2_7b_bootcamp.yaml
```

2. Copy below config into llama2_7b_bootcamp.yaml

```
defaults:
  - _self_
  - optional tp_overlap@model.ub_tp_comm_overlap_cfg:

hydra:
  searchpath:
    - file:///opt/NeMo/examples/nlp/language_modeling/conf

run:
  name: llama2_7b_bootcamp
  results_dir: ${base_results_dir}/${.name}
  time_limit: "0-01:30:00"
  dependency: "singleton"
trainer:
  num_nodes: 2
  devices: 8
  accelerator: gpu
  precision: bf16 #16-mixed for FP_16 enabled
  logger: False # logger provided by exp_manager
  enable_checkpointing: False
  use_distributed_sampler: False
  max_epochs: null
  max_steps: 300000 # consumed_samples = global_step * global_batch_size
  max_time: "05:23:30:00" # days:hours:minutes:seconds
  log_every_n_steps: 1
  val_check_interval: 2000
  limit_val_batches: 0
  limit_test_batches: 0
  accumulate_grad_batches: 1
  gradient_clip_val: 1.0
exp_manager:
  explicit_log_dir: ${training.run.results_dir}/results
  exp_dir: null
  name: megatron_llama
  create_wandb_logger: false
  wandb_logger_kwargs:
    project: nemo_llama_pretrain
    name: ${training.run.name}
  resume_if_exists: false
  resume_ignore_no_checkpoint: true
  create_checkpoint_callback: False
  checkpoint_callback_params:
    monitor: val_loss
    save_top_k: 10
    mode: min
    always_save_nemo: False # saves nemo file during validation, not implemented for model parallel
    save_nemo_on_train_end: False # not recommended when training large models on clusters with short time limits
    filename: 'megatron_llama--{val_loss:.2f}-{step}-{consumed_samples}'
    model_parallel_size: ${multiply:${training.model.tensor_model_parallel_size}, ${training.model.pipeline_model_parallel_size}}
  log_step_timing: True
  step_timing_kwargs:
    sync_cuda: True
    buffer_size: 5

model:
  mcore_gpt: true
  micro_batch_size: 1
  global_batch_size: 2048 #1024
  rampup_batch_size: null
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  virtual_pipeline_model_parallel_size: null
  encoder_seq_length: 4096
  max_position_embeddings: 4096
  num_layers: 32
  hidden_size: 4096
  ffn_hidden_size: 11008
  num_attention_heads: 32
  init_method_std: 0.01
  use_scaled_init_method: true
  hidden_dropout: 0.0
  attention_dropout: 0.0
  ffn_dropout: 0.0
  kv_channels: null
  apply_query_key_layer_scaling: true
  normalization: rmsnorm
  layernorm_epsilon: 1.0e-05
  do_layer_norm_weight_decay: false
  make_vocab_size_divisible_by: 128
  pre_process: true
  post_process: true
  persist_layer_norm: true
  bias: false
  activation: fast-swiglu
  headscale: false
  transformer_block_type: pre_ln
  openai_gelu: false
  normalize_attention_scores: true
  position_embedding_type: rope
  rotary_percentage: 1.0
  apply_rope_fusion: true
  attention_type: multihead
  share_embeddings_and_output_weights: false
  tokenizer:
    library: huggingface
    use_fast: true
    type: /data/tokenizer
    model: null
    delimiter: null
    vocab_file: null
    merge_file: null
  native_amp_init_scale: 4294967296
  native_amp_growth_interval: 1000
  hysteresis: 2
  fp32_residual_connection: false
  fp16_lm_cross_entropy: false
  megatron_amp_O2: true
  grad_allreduce_chunk_size_mb: 100
  grad_div_ar_fusion: true
  gradient_accumulation_fusion: true
  bias_activation_fusion: true
  bias_dropout_add_fusion: true
  masked_softmax_fusion: true
  seed: 1234
  resume_from_checkpoint: null
  use_cpu_initialization: false
  onnx_safe: false
  apex_transformer_log_level: 30
  gradient_as_bucket_view: true
  sync_batch_comm: false
  activations_checkpoint_granularity: null
  activations_checkpoint_method: block
  activations_checkpoint_num_layers: 0
  num_micro_batches_with_partial_activation_checkpoints: null
  activations_checkpoint_layers_per_pipeline: null
  sequence_parallel: false

  ## Transformer Engine
  # Refer to https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_base_model.py for param descriptions
  transformer_engine: true
  fp8: False # enables fp8 in TransformerLayer forward
  fp8_e4m3: False # sets fp8_format = recipe.Format.E4M3
  fp8_hybrid: True #False # sets fp8_format = recipe.Format.HYBRID
  fp8_margin: 0 # scaling margin
  fp8_interval: 1 # scaling update interval
  fp8_amax_history_len: 128 # Number of steps for which amax history is recorded per tensor
  fp8_amax_compute_algo: max # 'most_recent' or 'max'. Algorithm for computing amax from history
  fp8_params: True
  use_emha: False
  ub_tp_comm_overlap: False
  tp_comm_atomic_ag: False
  tp_comm_atomic_rs: False
  use_flash_attention: true
  distributed_adam_bucket_merge_size: 4
  optim:
    name: distributed_fused_adam
    lr: 0.0001
    weight_decay: 0.1
    betas:
      - 0.9
      - 0.95
    bucket_cap_mb: 400
    overlap_grad_sync: true
    overlap_param_sync: true
    contiguous_grad_buffer: true    
    contiguous_param_buffer: true   
    sched:
      name: CosineAnnealing
      warmup_steps: 500
      constant_steps: 0
      min_lr: 1e-5
  data:
    data_impl: mock #mmap
    splits_string: "90,8,2"
    seq_length: 4096
    skip_warmup: true
    num_workers: 4
    exchange_indices_distributed: true
    dataloader_type: single
    reset_position_ids: false
    reset_attention_mask: false
    eod_mask_loss: false
    index_mapping_dir: /data/trimmed/idx
    data_prefix:
    - 1.0
    - /data/trimmed/my-t5_00_text_document
```

3. Create a new training script called `train_llama2.sh` in the `launcher_scripts` directory.

- Note the new `llama2_7b_bootcamp` config being used for training. View `conf/config.yaml` to see the different workflows available through NeMo.
- To enable the workload to utilise fastrak networking, the host node's tcpxo binaries are mounted on the container 

```
#!/bin/bash

source ../nemo_env/bin/activate

MAX_STEPS=15
NUM_NODES=2

# Auth to access Llama HF repo
HF_TOKEN=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

python main.py \
    launcher_scripts_path=${PWD} \
    stages=[training] \
    training=llama/llama2_7b_bootcamp \
    env_vars.TRANSFORMERS_OFFLINE=0 \
    +env_vars.HF_TOKEN=${HF_TOKEN} \
    container=../nemofw+tcpxo-24.05.llama3.1.sqsh \
    container_mounts="['/var/lib/tcpxo/lib64',${PWD}/data:/data]" \
    cluster.srun_args=["--container-writable"] \
    training.trainer.max_steps=${MAX_STEPS} \
    training.trainer.num_nodes=${NUM_NODES} \
    training.trainer.val_check_interval=${MAX_STEPS}
```

4. Download the llama2 tokenizer from `launcher_scripts/data` directory
```
# From launcher_scripts/data directory
wget https://storage.googleapis.com/injae-download/tokenizer/llama2_tokenizer.tar && tar xvf llama2_tokenizer.tar

# you should now have tokenizer files
ls tokenizer/
merges.txt  special_tokens_map.json  tokenizer_config.json  tokenizer.json  vocab.json
```

5. Run the training script with `bash train_llama2.sh`. This will result in NeMo automatically generating and submitting a slurm job using the user provided parameters in the script, and the NeMo training config

```
# Generate and submit a NeMo training job
bash train_llama2.sh
```

6. Monitor the progress using Slurm command `watch squeue`, where `R` means running. If something has failed, the job will disappear from the queue automatically. Note that first job will take ~10 min to run as the A3 Mega nodes need to download the NCCL and RxDM container for Fastrak to work. You can verify this by SSH'ing into a A3 mega VM and using `watch docker images`.

- Useful Slurm commands
  - `squeue` shows current job queue
  - `sinfo` shows available queues to run jobs
  - `scancel <jobID>` cancels a scheduled or running job
  - `sacct -J <jobID>` shows historical info about previous job  

7. Monitor the generated log files in `results/llama2_7b_bootcamp` directory. The file names are `log-nemo-megatron-llama2_7b_bootcamp_<JobID>.out` and `log-nemo-megatron-llama2_7b_bootcamp_<JobID>.err`

Example output
```
~/hpc-toolkit/examples/machine-learning/a3-megagpu-8g/nemo-framework/launcher_scripts$ tail -f results/llama2_7b_bootcamp/log-nemo-megatron-llama2_7b_bootcamp_78.out 
        weight_decay: 0.1
    )
[NeMo I 2024-07-22 12:05:15 lr_scheduler:948] Scheduler "<nemo.core.optim.lr_scheduler.CosineAnnealing object at 0x14efe819d8a0>" 
    will be used during training (effective maximum steps = 15) - 
    Parameters : 
    (warmup_steps: 500
    constant_steps: 0
    min_lr: 1.0e-05
    max_steps: 15
    )
Epoch 0: :  47%|████▋     | 7/15 [06:19<07:13, reduced_train_loss=11.50, global_step=5.000, consumed_samples=12288.0, train_step_timing in s=49.50]
```
**Note: Connect to one of your A3 Mega VMs via SSH and run the nvitop tool (after activating the `nemo_env` virtual environment we created previously). This will allow you to see the real time GPU utilisation rates, which is useful for debugging or performance tuning eg. Low GPU memory utilisation rate likely means you have room to further increase performance! (or if your job is crashing with CUDA OOM)**

8. Note the `train_step_timing in s=` value, as that shows the training throughput for the job for time it takes per global step (and global batch size).

9. **[Make a copy of this google sheets template](https://docs.google.com/spreadsheets/d/1VDaQ9reMmWr9FHowzOy_0_Iwxkg1Bwo5vIPeb1yqXqA/edit?resourcekey=0-G0uKUN05DynsJBKkRCJUAg&gid=1344899973#gid=1344899973)** to calculate your MFU and tokens/GPU/sec

### What is happening as the job is submitted?

- To ensure performance through multi-NIC network through Fastrak, Slurm checks if the job has been submitted with 2 or more requested nodes as part of [Epilog stage](https://slurm.schedmd.com/prolog_epilog.html), in `/opt/apps/adm/slurm/scripts/rxdm`. This is similar to a start up script being run for every submitted Slurm job.
- If the job needs Fastrak enabled, Slurm automatically starts the 2 x necessary sidecar NCCL and RxDM containers to run alongside the user job. This works the same for jobs running on GKE pods.
- There is a one time pull of the NCCL and RxDM container images on A3 Mega nodes on the first submitted job. Subsequent jobs will use this cached container images for immediate job starts.
- The NeMo framework from Nvidia is tightly integrated with Slurm (inc. AWS and Azure), where example configs and scripts are automatically generated when a new Slurm job is submitted. Take a look through `launcher_scripts/conf/config.yaml` and `launcher_scripts/conf/` for preconfigured workflow examples. 

## Step 3: Performance Tune and set a High Score!

1. Modify your NeMo configurations and training parameters in `launcher_scripts/conf/training/llama/llama2_7b.yaml` to performance tune your job and increase the training throughput result.

- ikwak@ highscore: 16,082 tokens/sec/GPU with 77.71% MFU with 16 GPUs

[Make a copy of this benchmark template](https://docs.google.com/spreadsheets/d/1VDaQ9reMmWr9FHowzOy_0_Iwxkg1Bwo5vIPeb1yqXqA/edit?resourcekey=0-G0uKUN05DynsJBKkRCJUAg&gid=1344899973#gid=1344899973) and put in your train_step_timing value to see your throughput (Tokens/sec/GPU) and the corresponding MFU.

`Protip 1: Watch out! you may get CUDA OOM errors as you start modifying your training parameters. A hint is to start by lowering your micro_batch_size.`

`Protip 2: To identify how much headroom is available on your GPU for optimisation, SSH into one of your A3 mega VM and monitor metrics in real time such as GPU utilisation using nvitop. If you still have unused GPU memory, you can tune further eg. Increase micro_batch_size, bucket sizes, down quant precision type..`
