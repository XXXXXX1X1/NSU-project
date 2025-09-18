
"""
summarizer.py — генерация заголовка из текста с помощью модели mT5 (Hugging Face).
"""

import os
import argparse
import torch
from transformers import pipeline, AutoTokenizer


MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"


def process_text_file(input_path: str, output_path: str) -> None:
    """
    Считать текст, сгенерировать заголовок и сохранить результат.

    Args:
        input_path: путь к входному .txt файлу
        output_path: путь к выходному .txt файлу
    """
    try:
        if not os.path.exists(input_path):
            raise ValueError("File not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            raise ValueError("File is empty")

        device = 0 if torch.cuda.is_available() else -1

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, legacy=False)
        summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
            tokenizer=tokenizer,
            device=device
        )

        result = summarizer(
            text,
            max_new_tokens=15,
            min_length=5,
            do_sample=False,
            num_beams=8,
            no_repeat_ngram_size=3,
            length_penalty=1.1
        )

        headline = result[0]["summary_text"].strip()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(headline + "\n")

        print("Processing completed.")

    except Exception as e:
        raise ValueError(f"Error during summarization: {e}")


def main():
    """Main function with command line argument processing"""
    parser = argparse.ArgumentParser(description="Generate headline from text using mT5 model")
    parser.add_argument("input_file", help="Path to input .txt file")
    parser.add_argument("output_file", help="Path to output .txt file")

    args = parser.parse_args()
    process_text_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
