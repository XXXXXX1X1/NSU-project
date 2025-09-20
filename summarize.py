"""
summarizer.py — генерация заголовка из текста с помощью модели mT5 (Hugging Face).
"""

import os
import argparse
import torch
from transformers import pipeline, AutoTokenizer


def read_text(input_path: str) -> str:
    """Считать текст из файла и проверить, что он не пустой."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError("File is empty")
    if len(text.split()) < 3:
        raise ValueError("Too few words for a headline (need ≥ 3).")
    return text


def get_device(device_arg: str) -> int:
    """Вернуть индекс устройства для transformers: -1 (CPU) или 0 (CUDA)."""
    da = device_arg.lower()
    if da == "cpu":
        return -1
    if da in {"cuda", "auto"}:
        try:
            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
    return -1


def build_summarizer(model_name: str, device: int):
    """Создать пайплайн суммаризации с токенизатором и моделью."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)
    return pipeline("summarization", model=model_name, tokenizer=tokenizer, device=device)


def validate_lengths(max_length: int, min_length: int) -> None:
    """Проверить корректность параметров длины."""
    if min_length <= 0 or max_length <= 0:
        raise ValueError("min_length and max_length must be > 0.")
    if max_length < min_length:
        raise ValueError("max_length must be ≥ min_length.")
    if max_length > 60:
        raise ValueError("max_length is too large for a headline (≤ 60 recommended).")


def generate_headline(
    text: str,
    summarizer,
    *,
    max_length: int,
    min_length: int,
    num_beams: int = 8,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.1,
) -> str:
    """Сгенерировать заголовок по тексту."""
    validate_lengths(max_length, min_length)
    result = summarizer(
        text,
        max_length=max_length,      # <-- по заданию!
        min_length=min_length,      # <--
        do_sample=False,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
        truncation=True,
    )
    headline = result[0]["summary_text"].strip()
    if not headline:
        raise RuntimeError("Empty headline from model.")
    return headline


def write_text(output_path: str, text: str) -> None:
    """Записать текст в файл."""
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def process_text_file(
    input_path: str,
    output_path: str,
    *,
    model_name: str,
    max_length: int,
    min_length: int,
    device: str,
) -> None:
    """Считать текст, сгенерировать заголовок и сохранить результат."""
    try:
        text = read_text(input_path)
        device_idx = get_device(device)
        summarizer = build_summarizer(model_name, device_idx)
        headline = generate_headline(
            text,
            summarizer,
            max_length=max_length,
            min_length=min_length,
        )
        write_text(output_path, headline)
        print("Processing completed.")
    except Exception as e:
        raise ValueError(f"Error during summarization: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate headline from text using mT5 model")
    p.add_argument("input_file", help="Path to input .txt file")
    p.add_argument("output_file", help="Path to output .txt file")
    p.add_argument("--model", default="csebuetnlp/mT5_multilingual_XLSum",
                   help="HF model name")
    p.add_argument("--max-length", type=int, default=15, help="Headline max length")
    p.add_argument("--min-length", type=int, default=5, help="Headline min length")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Device selection")
    return p.parse_args()


def main():
    args = parse_args()
    process_text_file(
        args.input_file,
        args.output_file,
        model_name=args.model,
        max_length=args.max_length,
        min_length=args.min_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
