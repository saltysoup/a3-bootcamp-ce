# Instructions for pre-training llama2 7b using NeMo on GKE

# Use exactly 2 x A3 mega VMs per CE only, in asia-northeast1-b zone

## ***Objective***
### *By following this lab, you will be able to deploy a A3 GKE Slurm cluster to run a llama2 pre-training job using Nvidia NeMo framework*

## ***Prerequisites***

### *This user guide assumes that you are familiar with Kubernetes concepts such as pod, node, deployment, namespaces etc and are familiar with GKE concepts such as nodepools, autoscaling, and auto provisioning.*

## ***Provisioning a GKE cluster***

### *This section shows how to create a cluster which has two NodePools:*
- The default NodePool contains 3 nodes with e2-medium machine type.
- The a3-multi-nic NodePool contains 2 nodes with a3-megagpu-8g machine type (8 H100 GPU per node), and supports TCPXO and GKE multi-networking).
- Steps are extracted from latest External User Guide as of July 2024:
[GKE A3 plus VM User Guide (Public Preview Customers)](https://docs.google.com/document/d/1D5umT4-WDuNnYf3ieQ5SfLdmvGRPBQGLB662udzwz8I/edit?tab=t.0#heading=h.n4ytmbxt737h)


## Step 1: Setup Network (Create VPCs, subnets and firewall rules)
### *Set environment variables*
```
export PREFIX="<yourLdap>"
export REGION="asia-northeast1"
export ZONE="asia-northeast1-b"
export MTU=8896
export PROJECT="injae-sandbox-340804"
```
### Create VPC and subnets
```
for N in $(seq 1 8); do
  gcloud compute --project=${PROJECT} \
    networks create \
    ${PREFIX}-net-$N \
    --subnet-mode=custom \
    --mtu=${MTU}
  gcloud compute --project=${PROJECT} \
    networks subnets create \
    ${PREFIX?}-sub-$N \
    --network=${PREFIX?}-net-$N \
    --region=${REGION?} \
    --range=192.168.$N.0/24
  gcloud compute --project=${PROJECT} \
    firewall-rules create \
    ${PREFIX?}-internal-$N \
    --network=${PREFIX?}-net-$N \
    --action=ALLOW \
    --rules=tcp:0-65535,udp:0-65535,icmp \
    --source-ranges=192.168.0.0/16
done
```
> Note: Subnets are configured with /24 range, which has space for 256 IPs. You should explicitly choose the range that fits your needs. If your cluster will have more than 1K nodes, consider a range with more spaces (e.g /21 for 2048 IPs)

## Step 2: Create a GKE Cluster

```
export GKE_VERSION=1.28.10-gke.1148001
export CLUSTER_NAME="<yourLdap>"

gcloud --project ${PROJECT} beta container clusters create ${CLUSTER_NAME} --enable-dataplane-v2 --enable-ip-alias --region ${REGION} --node-locations ${ZONE} --enable-multi-networking --cluster-version ${GKE_VERSION} --no-enable-autoupgrade
```

> Note: `--region` by specifying this flag, the GKE cluster will be a regional cluster. Compared with zonal clusters (`--zone`), regional clusters have more default node quota (5k v.s. 1k), and have [additional advantages](https://cloud.google.com/kubernetes-engine/docs/concepts/regional-clusters).

## Step 3: Create a A3 Mega VM Nodepool
### *Set the EV*
```
export NODE_POOL_NAME="a3plus-multi-nic"
export MACHINE_TYPE="a3-megagpu-8g"
export NODE_COUNT=2
export ACCELERATOR_ARG="type=nvidia-h100-mega-80gb,count=8,gpu-driver-version=latest"
```
### *Create the GPU Pool*
```
gcloud beta container node-pools create ${NODE_POOL_NAME} --region ${REGION} --node-locations ${ZONE} --cluster ${CLUSTER_NAME} --project ${PROJECT} --no-enable-autoupgrade --accelerator ${ACCELERATOR_ARG} --machine-type ${MACHINE_TYPE} --num-nodes ${NODE_COUNT} --additional-node-network network=${PREFIX}-net-1,subnetwork=${PREFIX}-sub-1 --additional-node-network network=${PREFIX}-net-2,subnetwork=${PREFIX}-sub-2 --additional-node-network network=${PREFIX}-net-3,subnetwork=${PREFIX}-sub-3 --additional-node-network network=${PREFIX}-net-4,subnetwork=${PREFIX}-sub-4 --additional-node-network network=${PREFIX}-net-5,subnetwork=${PREFIX}-sub-5 --additional-node-network network=${PREFIX}-net-6,subnetwork=${PREFIX}-sub-6 --additional-node-network network=${PREFIX}-net-7,subnetwork=${PREFIX}-sub-7 --additional-node-network network=${PREFIX}-net-8,subnetwork=${PREFIX}-sub-8 --enable-gvnic --scopes "https://www.googleapis.com/auth/cloud-platform" --reservation-affinity=specific --reservation=a3mega-bootcamp
```
* `--scopes https://www.googleapis.com/auth/cloud-platform` sets the node instance's scope to be cloud-platform. The scope set here is for testing convenience. You may want to limit the scope to configure finer-grained credentials in practice.
* `--placement-policy`, `--reservation-affinity` and `--reservation` if you are using a reservation, specify these flags to pass the policy name and reservation into the nodepool.
* `--host-maintenance-interval=PERIODIC` if you are using gSC, specify this flag to create a nodepool with gSC pool instead of the general pool
* `--max-pods-per-node`  Specify the maximum number of Pods per node to optimize IP address allocation. By default, GKE allows up to 110 pods per node on standard GKE clusters. Limiting the maximum number of pods to save Pod IPs reserved for each node, and make the cluster have capability to host more nodes.
* `--max-surge-upgrade` If using `--reservation` and expecting to constantly utilize the entire reservation, be sure to tweak `--max-surge-upgrade` and `--max-unavailable-upgrade` based on your workload's tolerance. Surge upgrades attempt to create a new node when upgrading a node pool which will fail if the reservation is fully utilized, setting `--max-surge-upgrade=0 & --max-unavailable-upgrade={Number > 0}` would allow upgrades to proceed. Full  details for settings to help you choose values can be found at [GKE nodepool Surge Settings](https://cloud.google.com/kubernetes-engine/docs/concepts/node-pool-upgrade-strategies#surge-settings). 

## Step 4: Check the driver installer and gpu device plugin

```
# Check the driver installer and gpu device plugin
kubectl get pod -n kube-system
# Output
nvidia-gpu-device-plugin-large-cos-2z7p7     1/1     Running   0          4h12m
nvidia-gpu-device-plugin-large-cos-p4dn5     1/1     Running   0          4h12m
```
> Note: The GPU driver is by default installed with `ACCELERATOR_ARG="type=nvidia-h100-mega-80gb,count=8, gpu-driver-version=latest"` in the last step. Otherwise, apply the following command to manually install the driver.


## Step 5: Verify count of nvidia.com/gpu resources on node

```
# Verify count of nvidia.com/gpu resources on node
kubectl describe nodes
# Output.
Capacity:
  ...
  nvidia.com/gpu:             8
Allocatable:
  ...
  nvidia.com/gpu:             8
```

## ***Validating TCPXO networking on VMs through NCCL benchmark***

## Step 1: Running TCPXO with NCCL

This section shows how to install the TCPXO NCCL plugin and run a 2 node allgather NCCL test.

> Note: The TCPXO feature has two components:
>* The NCCL plugin installer runs as a Daemonset
>* The TCPXO daemon runs as a sidecar container together with the main GPU application container.

![](https://github.com/saltysoup/a3-bootcamp-ce/assets/12909178/3e6f3bf6-6a34-4a04-878d-ac4f971beff0)

### Step 2: Install the TCPXO NCCL plugin
The plugin will install a specific NCCL library and the TCPXO binary to the node. After applying the following daemonset manifest, you can find NCCL libraries, e.g. `libnccl.so`, `libnccl-tuner.so` and `libnccl-net.so`  in `/home/kubernetes/bin/nvidia/lib64` from the node’s underlying VM. 
These files will be by default mounted to `/usr/local/nvidia/lib64` together with NVIDIA related libraries in your main GPU application container.
```
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-tcpxo/nccl-tcpxo-installer.yaml
```
Wait until the nccl plugin is running, which takes ~120s.

```
kubectl get pod -n kube-system
# Output
nccl-tcpxo-installer-6c2pv                    1/1     Running   0          2m11s
nccl-tcpxo-installer-qgg82                    1/1     Running   0          2m11s
```
### Step 3: Deploying NCCL test workload
The following command deploys two pods with [nccl-test](https://github.com/NVIDIA/nccl-tests).

```
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-tcpxo/nccl-test.yaml
```

### Step 4: Configure the TCPXO daemon

The manifest you just deployed has two Pods. Each pod has two containers, one is the nccl-test container, and the other one is a **TCPXO daemon**. This is the management service which needs to run alongside the main GPU application container that intends to use TCPXO. 

> Note: **Every application run as a Pod needs to have this tcpxo daemon container as a sidecar with required volume mounts.**

Here is an example manifest of adding this daemon as a sidecar container into GPU workloads:

```
apiVersion: v1
kind: Pod
metadata:
  name: a3plus-workloads
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
...
  containers:
    - name: tcpxo-daemon
      image: us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/tcpgpudmarxd-dev:v1.0.9
      imagePullPolicy: Always
      command: ["/bin/sh", "-c"]
      args:
        - |
          set -ex
          chmod 755 /fts/entrypoint_rxdm_container.sh
          /fts/entrypoint_rxdm_container.sh --num_hops=2 --num_nics=8 --uid= --alsologtostderr
      securityContext:
        privileged: true
      volumeMounts:
        - name: nvidia-install-dir-host
          mountPath: /usr/local/nvidia
      env:
        - name: LD_LIBRARY_PATH
          value: /usr/local/nvidia/lib64
    - name: main-application-container
...
      env:
        - name: LD_LIBRARY_PATH
          value: /usr/local/nvidia/lib64
      securityContext:
        privileged: true
      resources:
        limits:
          nvidia.com/gpu: 8
  volumes:
    - name: nvidia-install-dir-host
      hostPath:
        path: /home/kubernetes/bin/nvidia
```
> - In this example: 
>    - The whole GPU application pod needs to use hostNetwork.
>    - The TCPXO sidecar needs to be in privileged mode.
>    - The main application container needs to be in privileged mode.

### Step 5: Run NCCL test workload

After deploying the NCCL test manifest, you’ll need to run the following commands to trigger a NCCL all gather for 2 nodes. 

```
kubectl exec --stdin --tty --container=nccl-test nccl-test-host-1 -- /scripts/allgather.sh nccl-host-1 nccl-host-2
```

Here is the example result for the nccl-test workload:

![](https://github.com/saltysoup/a3-bootcamp-ce/assets/12909178/ab9075de-6ea1-4b72-9edd-6fa4e3d1613c)

> -With above example, this shows an approx ~195-200 GB/s being used for message sizes from 1GB+ on the NCCL test. This shows usage of almost all available 1600 Gbps network bandwidth, using conversion formular (X GB/s * 8) = Y Gbps 

### *Optional : Topology Awareness Setup*

If you are using the compact placement or gSC shared reservation when creating A3+ nodepool, you can set up topology awareness configuration to gain better network performance.

**This is not available for this lab, but is highly recommended to enable for large scale GPU clusters.**

### Please check README.md on the external github for an example of how to take advantage of the performance boosts of this feature.
---

## ***Running Llama2 pre-training benchmark***

## Step 1: Deploy Llama2-7b training job via using helm:

This section shows how to configure and run the NeMo scripts using [Helm](https://helm.sh/). When the workload container starts, it will automatically run the NCCL plugin and RxDM sidecar that was previously configured to enable multi-NIC bandwidth on the training job. 

Take a look at the directory layout

```
ikwak@penguin:~/a3-bootcamp-ce/Day1/pretraining-with-nemo-on-gke$ tree nemo-on-k8s/
nemo-on-k8s/
├── docker
│   └── Dockerfile
└── helm-context
    ├── Chart.yaml
    ├── nemo-configurations
    │   ├── gpt-175b-fp8.yaml
    │   ├── gpt-175b.yaml
    │   ├── gpt-5b.yaml
    │   ├── llama2-7b-fp8.yaml
    │   ├── llama2-7b.yaml
    │   ├── llama3-70b.yaml
    │   ├── mixtral-8x7b-nvidia-configs.yaml
    │   └── mixtral8x7b.yaml
    ├── selected-configuration.yaml -> nemo-configurations/llama2-7b.yaml
    ├── templates
    │   └── nemo-example.yaml
    └── values.yaml

5 directories, 13 files
```

- In `docker/Dockerfile`, we're using the publicly available nemo container `nvcr.io/nvidia/nemo:24.05.llama3.1` and installing gcloud sdk to use gcsfuse in the container   

- In `helm-context`, there are 3 x significant yaml files worth noting:
  - In `nemo-configurations`, you can find example training configurations for NeMo. In this lab, we will be using `llama2-7b.yaml`
  - To allow easier switching between training configs, `selected-configuration.yaml` is used as symlink eg. `ln -sf nemo-configurations/llama2-7b.yaml selected-configuration.yaml`. This file is parsed when the training job is deployed. 
  - In `values.yaml`, 
  - In `templates/nemo-example.yaml`, you can find how the final kubernetes manifest for the job is created using the values from above yaml files 




```
helm install NAME-OF-YOUR-JOB helm-context/
```
You should see some output like this:
```
NAME: felix
LAST DEPLOYED: Mon Jul 22 01:14:14 2024
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
```
### Step 7.2: Check your training job
```










