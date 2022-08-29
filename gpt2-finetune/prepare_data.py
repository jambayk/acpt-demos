import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import requests

parser = argparse.ArgumentParser(description="Prepare data for Homer fine-tuning")

parser.add_argument("--source", type=str, default="homer_source.json", help="Json file with sources information")
parser.add_argument("--split", type=int, default=80, help="Percentage of data to use as training data")


def check(s, ignore):
    r = []
    for a in ignore.keys():
        if not hasattr(str, a):
            continue

        fn = getattr(str, a)
        if type(ignore[a]) == str:
            r += [fn(s, ignore[a])]

        elif type(ignore[a]) == list:
            r += [fn(s, i) for i in ignore[a]]

        elif type(ignore[a]) == bool and ignore[a]:
            r += [fn(s)]

    return any(r)


def substitute(s, replace):
    for a in replace.keys():
        s = re.sub(a, replace[a], s)
    return s


def load(
    title="",
    source="",
    start=0,
    end=100,
    ignore={},
    replace={},
    script_dir=None,
):
    print(f"Loading {title}")
    # get filename
    a = urlparse(source)
    file = os.path.basename(a.path)

    c = script_dir / ".cache"
    if not c.exists():
        os.makedirs(str(c))
    cfile = c.joinpath(file)
    if not cfile.exists():
        response = requests.get(source)
        with open(str(cfile), "wt", encoding="utf-8") as f:
            f.write(response.text)

    # load text
    with open(str(cfile), "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.encode("ascii", errors="ignore").decode("ascii").split("\n")[start:end]

    # cleaned sentences
    sentences = [
        f"{s.strip()}."
        for s in " ".join(
            [substitute(item, replace).strip() for item in lines if len(item) > 0 and not check(item, ignore)]
        ).split(".")
    ]
    print("done!")
    return sentences


def main(raw_args=None):
    args = parser.parse_args(raw_args)
    print(args.__dict__)

    assert args.split < 100, "Split value must be less than 100"
    train = args.split / 100.0
    print(f"Using {train} for training and {1-train} for validation")

    root_dir = Path(__file__).resolve().parent
    data_source = root_dir / args.source

    data_dir = root_dir / "code" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # clean data and create data files
    with open(str(data_dir / "training_raw.txt"), "w") as t:
        with open(str(data_dir / "validation_raw.txt"), "w") as v:
            with open(str(data_source), "r") as src:
                sources = json.load(src)

                for s in sources:
                    text = load(**s, script_dir=root_dir)
                    for i in range(len(text)):
                        if i < len(text) * train:
                            print(text[i], file=t)
                        else:
                            print(text[i], file=v)


if __name__ == "__main__":
    main()
