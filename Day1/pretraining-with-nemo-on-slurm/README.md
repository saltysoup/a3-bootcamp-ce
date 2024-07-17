# Instructions for Slurm

# Use exactly 2 x A3 mega VMs per CE only, in asia-northeast1-b zone

## ***Objective***
### *By following this document, you will be able to deploy a A3 plus Slurm cluster to run a llama2 pre-training job using Nvidia NeMo framework*

## ***Provisioning a cluster***

## Step 1: Use the public deployment guide, with minor alterations

**Note: It's recommended to not use cloudshell for the following, as session timeout can impact the long terraform deployment time**

```
project_id: injae-sandbox-340804
region: asia-northeast1
zone: asia-northeast1-b
```

1. Install the Cloud HPC toolkit in your local environment, including its dependencies [using this link.](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster)

1. [Follow the public deployment guide](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster) with the following changes:

1. Make changes before creating a new reservation with `--vm-count=2` in `--zone=asia-northeast1-b`
1. Make changes to the base deployment file with `network_name_system: yourUsername-net` and `subnetwork_name_system: yourUsername-subnet`
1. Make changes before deploying the VPC and Filestore in `slurm-a3mega-base.yaml`. Change the values to `filestore_tier: BASIC_SSD` and `size_gb: 2560` in line 53-54.
1. Make changes to the cluster deployment file `deployment-image-cluster.yaml`. Change the values for `network_name_system:`, `subnetwork_name_system:` and `a3mega_reservation_name:` using your respective values from above. As we are not using a gSC reservation, set `a3mega_maintenance_interval: ""`.
1. To save ~20 min, an custom OS image has already been built ahead of time, called `final_image_family: slurm-a3mega`. It's recommended to skip the [Build the custom OS image](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster#build-image) step
1. Provision the Slurm cluster. This will take approx 10 min to complete.


## ***Validate network performance on the cluster with a NCCL test***

## Step 1: Use the public documented guide

1. To perform a basic performance validation of the cluster, we'll be running a NCCL test across 2 x A3 Mega nodes. This will verify that multi-NIC performance using Fastrak is functional, which is critical for optimising distributed training.
1. SSH into the newly created Slurm login node using the provided `yourUsername@ikwak.altostrat.com` credentials.
1. Follow the guide on https://cloud.google.com/hpc-toolkit/docs/machine-learning/a3-mega-enable-gpudirect-tcpxo
1. You should get a result of ~180GB/s for 8589934592 message size (8GB) after running the NCCL all_reduce network test. This indicates that 1440 Gbps out of the maximum 1600 Gbps VM networking is being used.


## ***Run llama2 training job on NeMo***

## Step 1: Set up NeMo in your Slurm cluster

1. After SSH'ing to your Slurm login node, clone the HPC toolkit repo

```
cd ~
git clone https://github.com/GoogleCloudPlatform/hpc-toolkit.git && cd hpc-toolkit/examples/machine-learning/a3-megagpu-8g/nemo-framework
```

2. Update `Dockerfile` to use the latest NeMo container

```
ARG NEMOFW_VERSION=24.05
FROM nvcr.io/nvidia/nemo:${NEMOFW_VERSION}

ENV NCCL_FASTRAK_CTRL_DEV=enp0s12
ENV NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0
ENV NCCL_SOCKET_IFNAME=enp0s12
ENV NCCL_CROSS_NIC=0
ENV NCCL_ALGO=Ring
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

RUN echo "/var/lib/tcpxo/lib64" >> /etc/ld.so.conf.d/tcpxo.conf && ldconfig
ENV LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH
```

3. Update `setup_nemo.sh` to use the latest NeMo container

```
: "${NEMOFW_VERSION:=24.05}"

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

5. Install NeMo Framework required packages in a new python virtual environment
```
python3 -m venv nemo_env
source env/bin/activate
pip install -r requirements.txt # Copied from the NeMo Framework Container earlier
# This is needed to use 23.11 and python3.11, which is what is present on Debian 12.
# nvitop is a useful tool to real time monitor GPU utilisation and processes for debugging/optimisation 
pip install -U hydra-core nvitop
```

6. Create an example training job script for running a 5B paramater GPT3 model

```
cd launcher_scripts
mkdir -p data
```

7. Create a new script `train.sh`
```
#!/bin/bash

source ../nemo_env/bin/activate

MAX_STEPS=10
NUM_NODES=2

python main.py \
    launcher_scripts_path=${PWD} \
    stages=[training] \
    training=gpt3/5b \
    env_vars.TRANSFORMERS_OFFLINE=0 \
    container=../nemofw+tcpxo-24.05.sqsh \
    container_mounts='["/var/lib/tcpxo/lib64"]' \
    cluster.srun_args=["--container-writable"] \
    training.model.data.data_impl=mock \
    training.model.data.data_prefix=[] \
    training.trainer.max_steps=${MAX_STEPS} \
    training.trainer.val_check_interval=${MAX_STEPS} \
    training.trainer.limit_val_batches=0.0 \
    training.exp_manager.create_checkpoint_callback=False \
    training.exp_manager.resume_if_exists=False \
    training.trainer.num_nodes=${NUM_NODES}
```

8. Submit a new job by running `train.sh`. This will result in NeMo generating new training job scripts and submitting to Slurm for execution.
```
bash train.sh
```

9. Monitor the progress using Slurm command `watch squeue`. `R` means running. If something has failed, the job will disappear from the queue automatically. Note that first job will take ~10 min to run as the A3 Mega nodes need to download the NCCL and RxDM container for Fastrak to work. You can verify this by SSH'ing into a A3 mega VM and using `docker images`.

10. Check the generated .out and .err job logs under the `results/gpt3` directory.

### What is happening?

- To ensure performance through multi-NIC network through Fastrak, Slurm checks if the job has been submitted with 2 or more requested nodes as part of [Epilog stage](https://slurm.schedmd.com/prolog_epilog.html). This is similar to a start up script for every submitted job.
- If the job needs Fastrak enabled, Slurm automatically starts the 2 x necessary sidecar NCCL and RxDM containers to run alongside the user job. This works the same for jobs running on GKE pods.
- There is a one time pull of the NCCL and RxDM container images on A3 Mega nodes on the first submitted job. Subsequent jobs will use this cached container images for immediate job starts.
- The NeMo framework from Nvidia is tightly integrated with Slurm (inc. AWS and Azure), where example configs and scripts are automatically generated when a new Slurm job is submitted. Check out `launcher_scripts/conf/config.yaml` and `launcher_scripts/conf` for preconfigured examples. 
- In the GPT3_5b example job, `conf/training/gpt3/5b.yaml` is used for job hyperparameter configs

## Step 2: Run and optimise llama2 training job in your Slurm cluster

1. N

  