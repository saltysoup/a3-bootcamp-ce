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

2. [Follow the public deployment guide](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster) with the following changes:

3. A reservation has already been created for you (you don't need to create a new one). Use the `a3mega-bootcamp` reservation when updating the [cluster deployment file](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster#update-deployment) later

4. Make the following changes in the [base deployment file](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster#update-filestore-deployment)

- `deployment_name: yourUsername-base`
- `network_name_system: yourUsername-net`
- `subnetwork_name_system: yourUsername-subnet`

5. Before [deploying the VPC and Filestore](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster#setup-filestore), Make changes in `slurm-a3mega-base.yaml`. Change the values to `filestore_tier: BASIC_SSD` and `size_gb: 2560` in line 53-54.
- **Note the private ip address of your Filestore instance in the deployment output. You will need this later!**

6. For [Update the cluster deployment file](https://cloud.google.com/hpc-toolkit/docs/deploy/deploy-a3-mega-cluster#update-deployment) step, make the following changes in `deployment-image-cluster.yaml`.

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
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a3mega
#SBATCH --exclusive

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

5. Monitor the job using the following command. This should take approx. 15 min to complete and will show the Status as "R" or Running. The job will automatically disappeart from the queue once it is completed. 
```
squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
                 3    a3mega setup_ne ext_ikwa  R       1:03      1 ikwak-a3meganodese
```

6. If successful, you should now see `requirements.txt`, `launcher_scripts` and `auto_configurator` in your current directory

7. Install NeMo Framework required packages in a new python virtual environment
```
python3 -m venv nemo_env
source env/bin/activate
pip install -r requirements.txt # Copied from the NeMo Framework Container earlier
# This is needed to use 23.11 and python3.11, which is what is present on Debian 12.
# nvitop is a useful tool to real time monitor GPU utilisation and processes for debugging/optimisation 
pip install -U hydra-core nvitop
```

8. Create an example training job script for running a 5B paramater GPT3 model

```
cd launcher_scripts
mkdir -p data
```

9. Create a new script `train.sh`
```
#!/bin/bash

source ../nemo_env/bin/activate

MAX_STEPS=20
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

10. Submit a new job by running `train.sh`. This will result in NeMo generating new training job scripts and submitting to Slurm for execution.
```
bash train.sh
```

11. Monitor the progress using Slurm command `watch squeue`. `R` means running. If something has failed, the job will disappear from the queue automatically. Note that first job will take ~10 min to run as the A3 Mega nodes need to download the NCCL and RxDM container for Fastrak to work. You can verify this by SSH'ing into a A3 mega VM and using `docker images`.

12. Check the generated .out and .err job logs under the `results/gpt3` directory.

**Note: Connect to one of your A3 Mega VMs via SSH and run the nvitop tool (after activating the `nemo_env` virtual environment we created previously). This will allow you to see the real time GPU utilisation rates, which is useful for debugging or performance tuning eg. Low GPU memory utilisation rate likely means you have room to further increase performance!**

### What is happening?

- To ensure performance through multi-NIC network through Fastrak, Slurm checks if the job has been submitted with 2 or more requested nodes as part of [Epilog stage](https://slurm.schedmd.com/prolog_epilog.html). This is similar to a start up script for every submitted job.
- If the job needs Fastrak enabled, Slurm automatically starts the 2 x necessary sidecar NCCL and RxDM containers to run alongside the user job. This works the same for jobs running on GKE pods.
- There is a one time pull of the NCCL and RxDM container images on A3 Mega nodes on the first submitted job. Subsequent jobs will use this cached container images for immediate job starts.
- The NeMo framework from Nvidia is tightly integrated with Slurm (inc. AWS and Azure), where example configs and scripts are automatically generated when a new Slurm job is submitted. Check out `launcher_scripts/conf/config.yaml` and `launcher_scripts/conf` for preconfigured examples. 
- In the GPT3_5b example job, `conf/training/gpt3/5b.yaml` is used for job hyperparameter configs

## Step 2: Run and optimise llama2 training job in your Slurm cluster

In this step, we'll be running a llama2 training job. Note that the configurations are 

1. 
  