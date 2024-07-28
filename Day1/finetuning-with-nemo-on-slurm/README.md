# Instructions for fine-tuning llama3 8b using NeMo on Slurm

# Use exactly 2 x A3 High VMs (2 x nodes) per CE only

## ***Objective***
### *By following this lab, you will be able to fine-tune a llama3-8b model and learn how to optimise training hyperparameters to tune for faster performance*

## Why are we doing this on Slurm? (and not GKE or Vertex)

- TCP Direct not supported on A3 High on Vertex
- A3 Mega support with fastrak in preview end July
- Goal is to see how hyperparameters can impact training performance, which is agnostic
- Easier to monitor and debug in real time

## ***Connect to a A3 High VM Slurm cluster***

1. Log into the GCE Console in the lab project `injae-sandbox-340804` using your provided lab IAM user account.  

2. SSH into VM `slurm99-login-0smpeolo-001` using the OSLogin from the browser.

3. Navigate to the working directory for the lab.

```
cd /home/bootcamp/hpc-toolkit/examples/machine-learning/a3-highgpu-8g/nemo-framework/launcher_scripts
```

4. Make a copy of the reference script for submitting a fine-tuning job to NeMo and Slurm

```
cp finetune_llama3.sh <yourLdap>-finetune_llama3.sh
#eg. cp finetune_llama3.sh ikwak-finetune_llama3.sh
```

5. Have a look at your script

```
#!/bin/bash

# MAKE A COPY OF THIS SCRIPT eg. ikwak-finetune_llama3.sh

# DO NOT MODIFY ORIGINAL
if [[ $# -eq 0 ]] ; then
    echo 'Missing required job name eg. bash finetune_llama3.sh ikwak-1 '
    exit 1
fi

source ../nemo_env/bin/activate

MAX_STEPS=20
NUM_NODES=2
JOB_NAME=$1

# Auth to access Llama HF repo
HF_TOKEN=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

python main.py \
    launcher_scripts_path=${PWD} \
    stages=[fine_tuning] \
    fine_tuning=llama/squad \
    env_vars.TRANSFORMERS_OFFLINE=0 \
    +env_vars.HF_TOKEN=${HF_TOKEN} \
    container=../nemofw+tcpx-24.05.llama3.1.sqsh \
    container_mounts='["/var/lib/tcpx/lib64","/run/tcpx-\${SLURM_JOB_ID}:/run/tcpx","/mnt:/mnt"]' \
    cluster.srun_args=["--container-writable"] \
    fine_tuning.run.name=$JOB_NAME \
    fine_tuning.trainer.max_steps=${MAX_STEPS} \
    fine_tuning.trainer.num_nodes=${NUM_NODES} \
    fine_tuning.trainer.val_check_interval=${MAX_STEPS}
    # add your finetuning configs to optimise performance
```

- Requires a name to be passed as a parameter for running the job (this will make locating your log files easier later on)
- Activates the prepared python virtual environment, which has the necessary nemo python packages already installed
- Invokes NeMo's `main.py`, with minimal set of parameters to run our fine-tuning job

6. Make a copy of the NeMo fine-tuning job configuration. This yaml file describes model level configuration to run our fine-tuning job.

```
cp conf/fine_tuning/llama/squad.yaml conf/fine_tuning/llama/<yourLdap>>.yaml
#eg cp conf/fine_tuning/llama/squad.yaml conf/fine_tuning/llama/ikwak.yaml   
``` 

7. Update your NeMo job yaml to give a new `task_name`. This will make locating your log files easier later on.

```
# Open your NeMo job yaml
vim conf/fine_tuning/llama/<yourLdap>.yaml

# update the value for task_name: with your ldap
run:
  name: ${.task_name}_${.model_train_name}
  time_limit: "04:00:00"
  dependency: null #"singleton"
  convert_name: convert_nemo
  model_train_name: llama3_8b
  convert_dir: ${base_results_dir}/${fine_tuning.run.model_train_name}/${fine_tuning.run.convert_name}
  task_name: "ikwak"  # Rename this name to be more clear
  results_dir: ${base_results_dir}/${fine_tuning.run.model_train_name}/${fine_tuning.run.task_name}
``` 

8. Now, update your training script `yourLdap-finetune_llama3.sh` to reference your copy of the job configuration yaml in main.py's `fine_tuning` parameter
```
# Open the script
vim <yourLdap>-finetune_llama3.sh

# Your main.py should look similar to this
..
..
python main.py \
    launcher_scripts_path=${PWD} \
    stages=[fine_tuning] \
    fine_tuning=llama/ikwak \
    env_vars.TRANSFORMERS_OFFLINE=0 \
..
..
```

8. Submit a test job using the default (unoptimised) settings with `bash yourLdap-finetune_llama3.sh someJobName`

```
injae_ikwak_altostrat_com@slurmy99-login-0smpeolo-001:/home/bootcamp/hpc-toolkit/examples/machine-learning/a3-highgpu-8g/nemo-framework/launcher_scripts$ bash ikwak-finetune_llama3.sh test_1

/home/bootcamp/hpc-toolkit/examples/machine-learning/a3-highgpu-8g/nemo-framework/nemo_env/lib/python3.10/site-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.19) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
Job nemo-megatron-test_1 submission file created at '/home/bootcamp/hpc-toolkit/examples/machine-learning/a3-highgpu-8g/nemo-framework/launcher_scripts/results/llama3_8b/ikwak/nemo-megatron-test_1_submission.sh'
Job nemo-megatron-test_1 submitted with Job ID 213
```

- The fine-tuning job uses a llama3 checkpoint and tokenizer stored in the `launcher_scripts/data/llama_3_8b` directory. The fine-tuning job yaml refers to this with `restore_from_path: ${data_dir}/llama_3_8b`.
- In the first run, the job will automatically download example dataset for instruction tuning the llama3 model. This will take a few min to complete.
- Monitor the progress using Slurm command `watch squeue`, where R means running. If something has failed, the job will disappear from the queue automatically.

9. Verify that your job is successfully running via log files generated in `results/llama3_8b/<task_name>/log-nemo-megatron-<someJobName>_<SlurmJobID>.out` and `results/llama3_8b/<task_name>/log-nemo-megatron-<someJobName>_<SlurmJobID>.err`.

### Note the value for your job's train_step_timing in

This is the key information for determining training throughput. This indicates it takes xx.xx seconds for every step in the training pass. This can be used to measure and compare performance across clusters and models.

```
# This output shows train_step_timing of 14.6 seconds

(base) injae_ikwak_altostrat_com@slurmy99-login-0smpeolo-001:/home/bootcamp/hpc-toolkit/examples/machine-learning/a3-highgpu-8g/nemo-framework/launcher_scripts$ tail -f results/llama3_8b/ikwak/log-nemo-megatron-test_1_213.out 
    will be used during training (effective maximum steps = 20) - 
    Parameters : 
    (min_lr: 1.0e-08
    warmup_steps: 1000
    last_epoch: -1
    max_steps: 20
    )
Sanity Checking: |          | 0/? [00:00<?, ?it/s][NeMo I 2024-07-25 15:05:15 num_microbatches_calculator:86] setting number of micro-batches to constant 128
Sanity Checking DataLoader 0: 100%|██████████| 2/2 [00:23<00:00,  0.09it/s][NeMo I 2024-07-25 15:05:38 num_microbatches_calculator:86] setting number of micro-batches to constant 128
Epoch 0: :  65%|██████▌   | 13/20 [03:25<01:50, reduced_train_loss=3.850, global_step=12.00, consumed_samples=24576.0, train_step_timing in s=14.60]
```

- It's recommended to measure your step timing after 10-15 steps, as the initial steps take time to warm up and stabilise

10. Make a copy of [this benchmark template](https://docs.google.com/spreadsheets/d/1VDaQ9reMmWr9FHowzOy_0_Iwxkg1Bwo5vIPeb1yqXqA/edit?resourcekey=0-G0uKUN05DynsJBKkRCJUAg&gid=226799209#gid=226799209) and record your own results in the `Finetuning workload - Llama3` tab.

## ***I've got a need for speed! (Performance Tuning)***

1. Update your NeMo job yaml with new configuration to see if you can get faster training throughput (train_step_timing)

```
# Open your NeMo job yaml
vim conf/fine_tuning/llama/<yourLdap>.yaml

# Try to make changes with parallelism, micro_batch_size, quantization (precision), global_batch_size etc..
```

**Note: The full fine tuning config yaml can be found in launcher_scripts/data/llama_3_8b/model_config.yaml. However, as this is a shared lab, please do not modify this and customise your own respective NeMo job yaml only (even though it has limited customisation in comparison)**

2. TIP: SSH into a GPU node that your job is running in and monitor your GPU metrics in real time using tool such as `nvitop`. For example, this will show you how much GPU memory is being utilised, which is helpful to debug CUDA OOM or available headroom to further use your GPUs to drive more performance

```
# This shows the fine-tuning job is currently running on VMs slurmy99-a3-ghpc-0 and slurmy99-a3-ghpc-1. You should SSH into either of these whilst the job is running to run nvitop

(base) injae_ikwak_altostrat_com@slurmy99-login-0smpeolo-001:/home/bootcamp/hpc-toolkit/examples/machine-learning/a3-highgpu-8g/nemo-framework/launcher_scripts$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
               214        a3 nemo-meg injae_ik  R       0:02      2 slurmy99-a3-ghpc-[0-1]
```

- **ikwak@'s highscore - 10.6 train_step_timing for llama3-8b, 16 GPUs**

### You must use the same tokenizer, model and full fine-tuning only (no PEFT/LoRa)
