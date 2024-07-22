# Inference with JetStream and TPUs on GKE

> In this lab, you will learn how to deploy open LLMs (Gemma specifically) for serving with TPUs on GKE.

## Prerequisites

This lab assumes you already have two GKE clusters up and running with TPU v5e accelerators.

## (Optional) kubectl Context Creation

To connect to existing GKE clusters, you have to create ~/.kube/config contexts on your local machine. To do so, run the following. You have to replace __${CLUSTER_NAME}__ and __${REGION}__ with the corresponding value for your own environment.

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION}
```

You have to run the above command for both GKE clusters, you will see two cluster when running below. It will show you which one is selected as the CURRENT context.

```bash
kubectl config get-contexts
```

To change the CURRENT context, run the following.

```bash
kubectl config use-context ${CONTEXT_NAME_OF_TPU_CLUSTER}
```

### Gemma Access Request

Go to [Gemma Model Card](https://www.kaggle.com/models/google/gemma) to request access. (If it's not granted right away, please ask Minjae Kang(@minjkang) or Injae Kwak(@ikwak) for help.)

### Kaggle Access Token Creation

Since we will download the model checkpoint from [Kaggle](https://www.kaggle.com/), we need to prepare a Kaggle access token.

Go to [Settings](https://www.kaggle.com/settings) page, and click __Create New Token__ under __API__ section to generate your access token. A file named __kaggle.json__ (which contains your token) will be downloaded.

## Deploy Gemma with JetStream to TPU Cluster

Now it's time to deploy Gemma with JetStream to our clusters. As the previous lab, we'll use [gemma-7b-it](https://www.kaggle.com/models/google/gemma/maxText/7b-it) (Instruction Tuned Gemma 1 7B) for reference.

### Setting Kaggle Access Token as a k8s Secret

Given __kaggle.json__ file, run the following command to ingest your Kaggle Access Token to your cluster. Make sure set the path to your __kaggle.json__ correctly.

```bash
kubectl create secret generic kaggle-secret \
--from-file=${YOUR-PATH-TO-kaggle.json}
```
