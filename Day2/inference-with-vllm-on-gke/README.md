# Inference with vLLM on GKE

> In this lab, you will learn how to deploy open LLMs (Gemma specifically) for serving on GKE.

## Prerequisites

This lab assumes you already have two GKE clusters up and running with GPU accelerators. (One with L4s attached, the other with H100s attached.) If not, you have to create clusters first.

## (Optional) kubectl Context Creation

To connect to existing GKE clusters, you have to create ~/.kube/config contexts on your local machine. To do so, run the following. You have to replace __${CLUSTER_NAME}__ and __${REGION}__ with the corresponding value for your own environment.

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION}
```

You have to run the above command for both GKE clusters, you will see two cluster when running below. It will show you which one is selected as the CURRENT context.

```bash
kubectl config get-contexts
```

If you have set up contexts successfully you will see two rows like below.

![k8s-ctx](img/k8s-ctx.png)

To change the CURRENT context, run the following.

```bash
kubectl config use-context ${CONTEXT_NAME_TO_SWITCH}
```

Be sure to remember the above command, as we will deploy the same model on two different clusters.

## Setting HuggingFace Access Token as a Secret

Since we will download the model checkpoint from [HuggingFace](https://huggingface.co/), we need to prepare a HuggingFace access token.

### Gemma Access Request

Go to [Gemma Model Card](https://huggingface.co/google/gemma-7b-it) to request access. (If it's not granted right away, please ask Minjae Kang(@minjkang) or Injae Kwak(@ikwak) for help.)

### HF Access Token Creation

At the [settings](https://huggingface.co/settings/tokens) page, you can generate your access token. Make sure to set __Token type__ as __Read__ when you create the token.

### Setting K8s Secret

Run the following command after replacing __${YOUR_HF_ACCESS_TOKEN}__ with your own token.

```bash
export HF_TOKEN=${YOUR_HF_ACCESS_TOKEN} && \
kubectl create secret generic hf-secret \
--from-literal=hf_api_token=$HF_TOKEN \
--dry-run=client -o yaml | kubectl apply -f -
```

## Deploy Gemma with vLLM

Now it's time to deploy Gemma with vLLM to our clusters. We'll use [gemma-7b-it](https://huggingface.co/google/gemma-7b-it)(Instruction Tuned Gemma 1 7B) for reference.

### Deploy on the L4 Cluster

We will start with deploying on the cluster with L4s first. Make sure your L4 cluster is selected as the current context before proceeding.

#### k8s Manifest Creation
Let's create a k8s manifest named __gemma-vllm-l4.yaml__ for *Service* and *Deployment*. Then paste the following to the file.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-gemma-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gemma-server
  template:
    metadata:
      labels:
        app: gemma-server
        ai.gke.io/model: gemma-1.1-7b-it
        ai.gke.io/inference-server: vllm
        examples.ai.gke.io/source: user-guide
    spec:
      containers:
      - name: inference-server
        image: minjkang/a3-bootcamp-lab3:latest
        imagePullPolicy: Always
        resources:
          requests:
            cpu: "2"
            memory: "80Gi"
            ephemeral-storage: "25Gi"
            nvidia.com/gpu: 8
          limits:
            cpu: "2"
            memory: "80Gi"
            ephemeral-storage: "25Gi"
            nvidia.com/gpu: 8
        command: ["python3", "-m", "vllm.entrypoints.api_server"]
        args:
        - --model=$(MODEL_ID)
        - --tensor-parallel-size=2
        env:
        - name: MODEL_ID
          value: google/gemma-1.1-7b-it
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: hf_api_token
        volumeMounts:
        - mountPath: /dev/shm
          name: dshm
      volumes:
      - name: dshm
        emptyDir:
            medium: Memory
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: gemma-server
  type: ClusterIP
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
```

#### Apply k8s Manifest to the Cluster

Then, let's deploy it by running the following command.

```bash
kubectl apply -f gemma-vllm-l4.yaml
```

A Pod in the cluster downloads the model weights from Hugging Face using your access token and starts the serving engine.

Wait for the Deployment to be available:

```bash
kubectl wait --for=condition=Available --timeout=700s deployment/vllm-gemma-deployment
```

View the logs from the running Deployment:

```bash
kubectl logs -f -l app=gemma-server
```

The Deployment resource downloads the model data. This process can take a few minutes. Once you succeed, the output will be similar to the following:

```bash
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### Set Up k8s Port Forwarding for Testing

To send inference requests, let's set up port forwarding by running the following command.

```bash
kubectl port-forward service/llm-service 8000:8000
```

The output is similar to the following:

```bash
Forwarding from 127.0.0.1:8000 -> 8000
```

#### Send Inference Requests

Now we can send an inference request using the below python script. You can replace the prompt with your own one for testing.

```python
import requests

if __name__ == "__main__":
    url = 'http://127.0.0.1:8000/generate'
    
    user_prompt = "I'm new to coding. If you could only recommend one programming language to start with, what would it be and why?"

    req_body = {
        "prompt": "<start_of_turn>user\n${user_prompt}<end_of_turn>\n",
        "temperature": 0.90,
        "top_p": 1.0,
        "max_tokens": 128
    }

    x = requests.post(url, json=req_body)

    print(x.text)
```

The following output shows an example of the model response:

```bash
{"predictions":["Prompt:\n<start_of_turn>user\nI'm new to coding. If you could only recommend one programming language to start with, what would it be and why?<end_of_turn>\nOutput:\nPython is often recommended for beginners due to its clear, readable syntax, simple data types, and extensive libraries.\n\n**Here are some other reasons why Python is a great language for beginners:**\n\n* **Beginner-friendly syntax:** Python's syntax is similar to natural language, making it easy for beginners to understand and write code.\n* **Dynamic typing:** Python automatically figures out the type of data you are working with, eliminating the need for explicit declaration.\n* **Object-oriented:** Python supports object-oriented programming, which allows you to organize and reuse code.\n* **Large library:** Python has a vast library of"]}
```

#### Run Benchmark Test

Let's run a benchmark test with the following script. It will send same inference request multiple times and then calculate latency and throughput.

```python
```

#### Clean up

Let's clean up Gemma on L4 cluster by running the following command.

```bash
kubectl delete -f gemma-vllm-l4.yaml
```

### Deploy on the H100 Cluster


