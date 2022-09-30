import argparse
import datetime
import json
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    BuildContext,
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)
from azure.identity import DefaultAzureCredential


def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description="GPT2 Finetune AML job submission")

    # workspace
    parser.add_argument(
        "--ws_config",
        type=str,
        required=True,
        help="Workspace configuration. Path is absolute or relative to where script is called from",
    )

    # parse args
    args = parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    args = get_args(raw_args)

    root_dir = Path(__file__).resolve().parent
    component_dir = root_dir / "components"

    # connect to the workspace
    ws_config_path = root_dir / args.ws_config
    ml_client = MLClient.from_config(credential=DefaultAzureCredential(), path=ws_config_path)

    # code directory
    code_dir = component_dir / "deploy-code"
    environment_dir = component_dir / "environment"

    # Creating a unique endpoint name with current datetime to avoid conflicts
    # online_endpoint_name = "acpt-demo-endpoint-" + datetime.datetime.now().strftime("%m%d%H%M%f")
    online_endpoint_name = "acpt-demo-endpoint-jk"

    # create an online endpoint
    print("Creating online endpoint...")
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name, description="GPT2 Finetuned Online Endpoint", auth_mode="key"
    )
    ml_client.begin_create_or_update(endpoint)

    # create a blue deployment
    print("Creating blue deployment...")
    blue_deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=online_endpoint_name,
        model="azureml:acpt-gpt2:1",
        environment=Environment(
            description="ACPT GPT2 fine-tune environment", build=BuildContext(path=environment_dir)
        ),
        code_configuration=CodeConfiguration(code=code_dir, scoring_script="score.py"),
        instance_type="Standard_DS5_v2",
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(blue_deployment)

    # blue deployment takes 100 traffic
    endpoint.traffic = {"blue": 100}
    ml_client.begin_create_or_update(endpoint)

    # # test the blue deployment with some sample data
    print("Testing blue deployment...")
    output = ml_client.online_endpoints.invoke(
        endpoint_name=online_endpoint_name,
        deployment_name="blue",
        request_file=root_dir / "sample_prompt.json",
    )

    sample_prompt = json.load(open(root_dir / "sample_prompt.json"))["prompt"]
    model_output = json.loads(output)["generated"][0]
    print(f"Prompt: {sample_prompt}")
    print(f"Model Output: {model_output}")


if __name__ == "__main__":
    main()
