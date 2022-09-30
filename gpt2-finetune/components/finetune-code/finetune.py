import argparse
import math
from itertools import chain
from pathlib import Path

from azureml.core.run import Run
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, default_data_collator


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# load dataset
def load_raw_dataset(train_file, validation_file, cache_dir=".cache"):
    data_files = {}
    dataset_args = {}
    data_files["train"] = train_file
    data_files["validation"] = validation_file

    extension = train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = True

    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir, **dataset_args)

    return raw_datasets


# helper function for grouping text into block size chunks
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the
    # model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]  # noqa: E203
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_and_batch_datasets(tokenizer, raw_datasets, block_size_):
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    global block_size
    block_size = block_size_
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    return train_dataset, eval_dataset


def main(
    model_path: str,
    tokenizer_path: str,
    config_path: str,
    train_path: str,
    validation_path: str,
    block_size: int,
    batch_size: int,
    max_steps: int,
    ort: bool,
    fp16: bool,
    deepspeed: bool,
):

    import os

    rank = os.environ.get("RANK", -1)
    assert rank != -1
    cache_dir = f".cache_{rank}"

    # get raw datasets
    raw_datasets = load_raw_dataset(train_path, validation_path, cache_dir=cache_dir)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast_tokenizer=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config_path)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset, eval_dataset = tokenize_and_batch_datasets(tokenizer, raw_datasets, block_size)

    training_args_dict = {
        "output_dir": ".outputs",
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": 4,
        "eval_accumulation_steps": 1,
        "max_steps": max_steps,
        "save_strategy": "no",
        "report_to": "azure_ml",
        "fp16": fp16,
        "deepspeed": "ds_config_zero_1.json" if deepspeed else None,
        "learning_rate": 5e-5,
    }

    # initialize training arguments
    training_args = TrainingArguments(**training_args_dict)

    if ort:
        from optimum.onnxruntime import ORTTrainer

        trainer_class = ORTTrainer
    else:
        from transformers import Trainer

        trainer_class = Trainer
    # Initialize Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
    )

    last_checkpoint = None

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_metrics)

    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(eval_dataset)
    perplexity = math.exp(eval_metrics["eval_loss"])
    eval_metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", eval_metrics)

    # generate text from prefix after fine-tuning
    from transformers import TextGenerationPipeline

    device = -1 if model.device.type == "cpu" else model.device.index
    text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)
    print(text_generator("The war in")[0]["generated_text"])
    print(text_generator("The market in America")[0]["generated_text"])

    if int(rank) == 0:
        # save trained config, tokenizer and model
        trained_model_folder = "model"
        trained_model_path = Path(trained_model_folder)
        trained_model_path.mkdir(parents=True, exist_ok=True)
        model.config.save_pretrained(trained_model_path / "config")
        tokenizer.save_pretrained(trained_model_path / "tokenizer")
        model.save_pretrained(trained_model_path / "weights")

        # register model
        run = Run.get_context()
        run.upload_folder(name="model", path=trained_model_folder)
        run.register_model(model_name="acpt-gpt2", model_path=trained_model_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT2 Fine-Tuning")

    parser.add_argument(
        "--model_path", type=str, default="gpt2", help="The model checkpoint for weights initialization"
    )
    parser.add_argument("--tokenizer_path", type=str, default="gpt2", help="Pretrained tokenizer path")
    parser.add_argument("--config_path", type=str, default="gpt2", help="Pretrained model configuration")

    parser.add_argument("--train_path", type=str, default="data/training_raw.txt", help="Pre-processed training data")
    parser.add_argument(
        "--validation_path", type=str, default="data/validation_raw.txt", help="Pre-processed validation data"
    )

    parser.add_argument("--block_size", type=int, default=1024, help="Block size for text in each training example")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step on each device")
    parser.add_argument("--max_steps", type=int, default=200, help="Max step that a model will run")

    parser.add_argument("--ort", type=str2bool, default=False, help="Use ORTModule")
    parser.add_argument("--fp16", type=str2bool, default=False, help="Use mixed precision")
    parser.add_argument("--deepspeed", type=str2bool, default=False, help="Use deepspeed")

    args = parser.parse_args()

    print(f"input parameters {vars(args)}")

    main(**vars(args))
