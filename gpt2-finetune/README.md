# GPT2 Fine-tuning Demo

This demo will show how to use ACPT along with accelerators such as onnxruntime training (through ORTModule) and DeepSpeed to fine-tune a GPT-2 model on Homer texts. 

## Background

GPT-2 is a transformers based language model that has been pre-trained on a large corpus of text data. It can be fine-tuned for task such as causal language modeling where it generates additional text based on prefix text by training it on additional data. 

In this demo, we will fine-tune it using Homer's The Illiad and The Odyssey so that it generates text that sound like Homer. We will use ACPT to create our training environment and leverage some of the training acceleration technologies it offers. 

## Set up

### AzureML 
The demo will be run on AzureML. Please complete the following prerequisites:

#### Local environment
Set up your local environment with azureml dependency for script submission:

```
pip install azure-ai-ml
```

#### AzureML Workspace
- An AzureML workspace is required to run this demo. Download the config.json file ([How to get config.json file from Azure Portal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace)) for your workspace. 
- The workspace should have a gpu cluster. This demo was tested with GPU cluster of SKU [Standard_ND40rs_v2](https://docs.microsoft.com/en-us/azure/virtual-machines/ndv2-series). See this document for [creating gpu cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python). We do not recommend running this demo on `NC` series VMs which uses old architecture (K80).

### Prepare Data
Prepare the data by running the following script which downloads the Homer texts and processes them into data files ready for training.
```
python prepare_data.py
```

## Run Experiments
The demo is ready to be run. 

`aml_submit.py` submits an AML job. This job builds the training environment and runs the fine-tuning script in it.

```bash
python aml_submit.py --ws_config [Path to workspace config json] --compute [Name of gpu cluster] --run_config [Accelerator configuration]
```

Here're the different configs and description that `job_submitter.py` takes through `--run_config` parameter.

| Config    | Description |
|-----------|-------------|
| no_acc    | PyTorch mixed precision (Default) | 
| ort       | ORTModule mixed precision |
| ds        | PyTorch + Deepspeed stage 1 |
| ds_ort    | ORTModule + Deepspeed stage 1|

An example job submission to a compute target named `v100-32gb-eus` and using ORTModule + Deepspeed:

```
python job_submitter.py --ws_config ws_config.json --compute v100-32gb-eus \
    --run_config ds_ort
```

Other parameters. 

| Name                | Description |
|---------------------|-------------|
| --nnode             | Number of nodes. Defaults to 1. |
| --nproc_per_node    | Number of processes per node. Defaults to 8. Set this value to the number of GPUs on each node. |
| --block_size        | Size of text blocks for each example. Defaults to 1024. | 
| --batch_size        | Model batchsize per GPU. Defaults to 8. |
| --max_steps         | Max step that a model will run. Defaults to 2000. |


## FAQ
### Problem with Azure Authentication
If there's an Azure authentication issue, install Azure CLI [here](https://docs.microsoft.com/en-us/cli/azure/) and run `az login --use-device-code`

### In case of `RuntimeError: CUDA out of memory` error
The issue is most likely caused by hitting a HW limitation on the target, this can be mitigated by using the following switches

`--batch_size` - Change to smaller batch size. Default is 8.
