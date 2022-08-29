import argparse
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import DefaultAzureCredential


def run_config_to_args(run_config):
    mapping = {
        "no_acc": ["--fp16", "True"],
        "ds": ["--fp16", "True", "--deepspeed", "True"],
        "ort": ["--fp16", "True", "--ort", "True"],
        "ds_ort": ["--fp16", "True", "--deepspeed", "True", "--ort", "True"],
    }
    return mapping[run_config]


def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description="GPT2 Finetune AML job submission")

    # workspace
    parser.add_argument(
        "--ws_config",
        type=str,
        required=True,
        help="Workspace configuration. Path is absolute or relative to where script is called from",
    )
    parser.add_argument("--compute", type=str, required=True, help="Compute target to run job on")

    # distributed training config
    parser.add_argument("--nnode", type=int, default=1, help="No of nodes. Default is 1")
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=8,
        help="No of GPUs per node. Default is 8",
    )

    # fine-tune hyperparameters
    parser.add_argument("--block_size", type=int, default=1024, help="Block size for text in each training example")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step on each device")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max step that a model will run")

    # accelerator hyperparameters
    parser.add_argument("--run_config", type=run_config_to_args, default="no_acc", help="Configs to run for model")

    # parse args, extra_args used for job configuration
    args = parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    args = get_args(raw_args)

    root_dir = Path(__file__).resolve().parent

    # connect to the workspace
    ws_config_path = root_dir / args.ws_config
    ml_client = MLClient.from_config(credential=DefaultAzureCredential(), path=ws_config_path)

    # code directory
    code_dir = root_dir / "code"
    environment_dir = root_dir / "environment"

    # tags
    tags = {"__nnode": args.nnode, "__nproc_per_node": args.nproc_per_node}

    # define the command
    command_job = command(
        description="ACPT GPT2 Finetune",
        display_name=f"gpt-finetune-{args.nnode}-{args.nproc_per_node}",
        experiment_name="acpt-gpt2-finetune",
        code=code_dir,
        command=(
            "python finetune.py"
            f" --block_size {args.block_size}"
            f" --batch_size {args.batch_size}"
            f" --max_steps {args.max_steps}"
            f" {' '.join(args.run_config)}"
        ),
        environment=Environment(
            description="ACPT GPT2 fine-tune environment", build=BuildContext(path=environment_dir)
        ),
        distribution={
            "type": "pytorch",
            "process_count_per_instance": args.nproc_per_node,
        },
        compute=args.compute,
        instance_count=args.nnode,
        tags=tags,
    )

    # submit the command
    print("submitting job")
    returned_job = ml_client.jobs.create_or_update(command_job)
    print("submitted job")

    aml_url = returned_job.studio_url
    print("job link:", aml_url)


if __name__ == "__main__":
    main()
