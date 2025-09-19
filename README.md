

---

# Text Summarization CLI

This script generates a short summary using Hugging Face Transformers’ `pipeline('summarization')` and prints it to stdout (optionally saves to a file). Fixed generation lengths: `max_length=15`, `min_length=5` (in tokens).

## Requirements

* Python 3.x
* Install dependencies via:

```bash
pip install -r requirements.txt
```

## Usage

* Run the script from the command line with the required arguments.

## Command-Line Arguments:

* `--input`: Path to the input `.txt` file (required, str).
* `--output`: Path to the output file to save the summary (optional, str).
* `--model`: Hugging Face model name (optional, str, default: `facebook/bart-large-cnn`).
* `--cpu`: Flag to force CPU even if CUDA is available (optional, no value needed).

## Example:

```bash
python summarize.py --input input.txt --output headline.txt
```
