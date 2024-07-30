# Inference with JetStream and TPUs on GKE

> In this lab, you will learn how to deploy open LLMs (Gemma specifically) for serving with TPUs on GKE.

Use the existing GKE clusters below

For Inference with vLLM on GKE, use cluster l4andvllm in region asia-southeast1 and cluster h100onvllm in region us-east5
For Inference with JetStream and TPUs on GKE, use cluster tpu-cluster-netherlands in region europe-west4

## Prerequisites

This lab assumes you already have two GKE clusters up and running with TPU v5e accelerators.

## Create a GCS Bucket

A GCS bucket is required to run this lab. You can either create one or use any existing bucket. Whichever you choose, please make sure your __k8s cluster's service account__ can access the bucket.

## (Optional) Create kubectl Context

To connect to existing GKE clusters, you have to create ~/.kube/config contexts on your local machine. To do so, run the following. You have to replace __${CLUSTER_NAME}__ and __${REGION}__ with the corresponding value for your own environment.

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION}
```

If you have to run the above command successfully, you will the context for your TPU v5e cluster from the output of the following command:

```bash
kubectl config get-contexts
```

To set your TPU v5e cluster as the CURRENT context, run the following.

```bash
kubectl config use-context ${CONTEXT_NAME_OF_TPU_CLUSTER}
```

### Request Gemma Access

Visit [Gemma Model Card](https://www.kaggle.com/models/google/gemma) to request access (If it's not granted right away, please ask Minjae Kang (minjkang@) or Injae Kwak (ikwak@) for help).

### Create Kaggle Access Token

Since we will download the model checkpoint from [Kaggle](https://www.kaggle.com/), we need to prepare a Kaggle access token.

Go to [Settings](https://www.kaggle.com/settings) page, and click __Create New Token__ under __API__ section to generate your access token. A file named __kaggle.json__ (which contains your token) will be downloaded.

## Deploy Gemma with JetStream to TPU Cluster

Now it's time to deploy Gemma with JetStream to our clusters. As the previous lab, we'll use [gemma-7b-it](https://www.kaggle.com/models/google/gemma/maxText/7b-it) (Instruction Tuned Gemma 1 7B) for reference.

### Set Kaggle Access Token as a k8s Secret

Given __kaggle.json__ file, run the following command to ingest your Kaggle Access Token to your cluster. Make sure set the path to your __kaggle.json__ correctly. Update the string ${YOUR_LDAP} with your LDAP to make it unique.

```bash
# Update the string ${YOUR_LDAP} with your LDAP to make it unique
kubectl create secret generic ${YOUR_LDAP}-kaggle-secret \
--from-file=${YOUR-PATH-TO-kaggle.json}
```

### Convert Checkpoint for MaxText and JetStream

> Even though Kaggle provides checkpoints for MaxText, it is not actually MaxText compatible. Instead, it is [Orbax](https://github.com/google/orbax) compatible.

We will convert the Kaggle-provided checkpoint to a MaxText compatible format first, and then convert it once again into unscanned one, which is used for JetStream model serving.

Follow the below instructions to download and convert the Gemma 7B model checkpoint files.

1. Create the following manifest as __job-7b.yaml__. Make sure to replace __${BUCKET_NAME}__ with your own bucket's name and __${YOUR_LDAP}__ with your LDAP.


  ```yaml
  apiVersion: batch/v1
  kind: Job
  metadata:
    name: ${YOUR_LDAP}-data-loader-7b # replace
  spec:
    ttlSecondsAfterFinished: 30
    template:
      spec:
        restartPolicy: Never
        containers:
        - name: ${YOUR_LDAP}-inference-checkpoint # replace
          image: us-docker.pkg.dev/cloud-tpu-images/inference/inference-checkpoint:v0.2.3
          args:
          - -b=${BUCKET_NAME} # replace
          - -m=google/gemma/maxtext/7b-it/2
          volumeMounts:
          - mountPath: "/kaggle/"
            name: kaggle-credentials
            readOnly: true
          resources:
            requests:
              google.com/tpu: 8
            limits:
              google.com/tpu: 8
        nodeSelector:
          cloud.google.com/gke-tpu-topology: 2x4
          cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice
        volumes:
        - name: kaggle-credentials
          secret:
            defaultMode: 0400
            secretName: ${YOUR_LDAP}-kaggle-secret # replace
  ```

2. Apply the manifest:

  ```bash
  kubectl apply -f job-7b.yaml
  ```

3. View the logs from the Job:

  ```bash
  kubectl logs -f jobs/${YOUR_LDAP}-data-loader-7b # replace
  ```

  When the Job is completed, the output is similar to the following:

  ```text
  Successfully generated decode checkpoint at: gs://${BUCKET_NAME}/final/unscanned/gemma_7b-it/0/checkpoints/0/items
  + echo -e '\nCompleted unscanning checkpoint to gs://${BUCKET_NAME}/final/unscanned/gemma_7b-it/0/checkpoints/0/items'

  Completed unscanning checkpoint to gs://${BUCKET_NAME}/final/unscanned/gemma_7b-it/0/checkpoints/0/items  
  ```

### Deploy JetStream

Now you will see checkpoint files created in your bucket. With those file, you will deploy the JetStream container to serve the Gemma model.

Follow below instructions to deploy the Gemma 7B instruction tuned model.

1. Create the following __jetstream-gemma-deployment.yaml__ manifest.  Make sure to replace __${BUCKET_NAME}__ with your own bucket's name and __${YOUR_LDAP}__ with your LDAP.

  ```yaml
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: ${YOUR_LDAP}-maxengine-server # replace
  spec:
    replicas: 1
    selector:
      matchLabels:
        app: ${YOUR_LDAP}-maxengine-server # replace
    template:
      metadata:
        labels:
          app: ${YOUR_LDAP}-maxengine-server # replace
      spec:
        nodeSelector:
          cloud.google.com/gke-tpu-topology: 2x4
          cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice
        containers:
        - name: ${YOUR_LDAP}-maxengine-server
          image: us-docker.pkg.dev/cloud-tpu-images/inference/maxengine-server:v0.2.2
          args:
          - model_name=gemma-7b
          - tokenizer_path=assets/tokenizer.gemma
          - per_device_batch_size=4
          - max_prefill_predict_length=1024
          - max_target_length=2048
          - async_checkpointing=false
          - ici_fsdp_parallelism=1
          - ici_autoregressive_parallelism=-1
          - ici_tensor_parallelism=1
          - scan_layers=false
          - weight_dtype=bfloat16
          - load_parameters_path=gs://${BUCKET_NAME}/final/unscanned/gemma_7b-it/0/checkpoints/0/items # replace
          ports:
          - containerPort: 9000
          resources:
            requests:
              google.com/tpu: 8
            limits:
              google.com/tpu: 8
        - name: jetstream-http # replace
          image: minjkang/a3-bootcamp-lab4:latest
          ports:
          - containerPort: 8000
  ---
  apiVersion: v1
  kind: Service
  metadata:
    name: ${YOUR_LDAP}-jetstream-svc # replace
  spec:
    selector:
      app: ${YOUR_LDAP}-maxengine-server # replace
    ports:
    - protocol: TCP
      name: jetstream-http
      port: 8000
      targetPort: 8000
    - protocol: TCP
      name: jetstream-grpc
      port: 9000
      targetPort: 9000
  ```

  The manifest sets the following key properties:

  - tokenizer_path: the path to your model's tokenizer.
  - load_parameters_path: the path in the Cloud Storage bucket where your checkpoints are stored.
  - per_device_batch_size: the decoding batch size per device, where one TPU chip equals one device.
  - max_prefill_predict_length: the maximum length for the prefill when doing autoregression.
  - max_target_length: the maximum sequence length.
  - model_name: the model name (gemma-7b).
  ici_fsdp_parallelism: the number of shards for fully sharded data parallelism (FSDP).
  - ici_tensor_parallelism: the number of shards for tensor parallelism.
  - ici_autoregressive_parallelism: the number of shards for autoregressive parallelism.
  - scan_layers: scan layers boolean flag (boolean).
  - weight_dtype: the weight data type (bfloat16).

2. Apply the manifest:

  ```bash
  kubectl apply -f jetstream-gemma-deployment.yaml
  ```

3. Verify the Deployment:

  ```bash
  kubectl get deployment
  ```

  The output is similar to the following:

  ```text
  NAME                              READY   UP-TO-DATE   AVAILABLE   AGE
  ${YOUR_LDAP}-maxengine-server     1/1     1            1           10m
  ```

4. View the HTTP server logs to check that the model has been loaded and compiled. It may take the server a few minutes to complete this operation.

  ```bash
  kubectl logs deploy/${YOUR_LDAP}-maxengine-server -f -c jetstream-http # replace
  ```

  The output is similar to the following:

  ```text
  INFO:     Started server process [1]
  INFO:     Waiting for application startup.
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
  ```

5. View the MaxEngine logs and verify that the compilation is done.

  ```bash
  kubectl logs deploy/${YOUR_LDAP}-maxengine-server -f -c maxengine-server # replace
  ```

  When everything is up and running, the output will be similar to the following:

  ```text
  DEBUG:jax._src.dispatch:Finished XLA compilation of jit(initialize) in 0.18877363204956055 sec
  2024-07-27 07:09:57,978 - jax._src.compiler - DEBUG - Not writing persistent cache entry for 'jit_initialize' because it took < 1.00 seconds to compile (0.19s)
  2024-07-27 07:09:57,978 - jax._src.dispatch - DEBUG - Finished XLA compilation of jit(initialize) in 0.18877363204956055 sec
  2024-07-27 07:09:58,039 - root - INFO - ---------Generate params 0 loaded.---------
  INFO:root:---------Generate params 0 loaded.---------
  ```

#### Set Up k8s Port Forwarding for Testing

To send inference requests, let's set up port forwarding by running the following command.

```bash
kubectl port-forward svc/${YOUR_LDAP}-jetstream-svc 8000:8000 # replace
```

The output is similar to the following:

```bash
Forwarding from 127.0.0.1:8000 -> 8000
```

### Send Inference Requests

Now we can send inference requests using __../inference.py__. You can replace *user_prompt* with your own one for testing.

```bash
python ../inference.py
```

The following output shows an example of the model response:

```json
{
    "prediction": "**Answer:**\n\nPython is an excellent choice for beginners due to its simple syntax, readability...",
    "benchmark": {
        "total_elapsed_time": 5.143557029998192,
        "total_tokens_generated": 279,
        "throughput": 54.242618167314866
    }
}
```

#### Run Benchmark Test

Let's run a benchmark test using __../benchmark.py__. It will send multiple inference requests concurrently and then calculate the average latency and throughput.

```bash
python ../benchmark.py
```

The output will be like:

```text
===== Result =====
Request Count: 30
Total Elapsed Time for Generation: 1294.31 seconds
Total Generated Tokens: 4490
Average Throughput: 3.47 tokens/sec
```

> With that, convert Average Throughput(tokens/sec) into Average Per Cost Performance(tokens/$) and Cost for 1M Tokens Generation($/1M tokens).
<!-- -->
> Compare this with L4 & H100 benchmark results. Do they differ significantly?

## Optimize Performance for Gemma with JetStream and TPUs on GKE

Now it's time to get your hands dirty. Your goal is to find an optimal setting(including both infrastructure options and MaxText+JetStream configurations) for Gemma to achieve minimum Cost for 1M Tokens Generation($/1M tokens).

> Note: Settings provided in this lab is far from the optimal👻.

You can use the existing k8s manifest and benchmarking script for your own experimentation. You can either add or modify MaxText+JetStream  configuration arguments. Share your results through the leaderboard.

> Hint: Refer to k8s manifest and notice how we have passed MaxText+JetStream settings!
<!-- -->
> Hint: See this [link](https://github.com/google/JetStream/blob/main/docs/online-inference-with-maxtext-engine.md#jetstream-maxtext-server-flag-descriptions) to get information about JetStream+MaxText setting arguments.

#### Clean up

1. Delete Gemma deployment on the cluster by running the following command.

  ```bash
  kubectl delete -f jetstream-gemma-deployment.yaml
  ```

2. Delete the bucket you've created to save checkpoints.
