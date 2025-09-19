
"""
summarize.py — краткое резюме текста без промптов и без пост-обрезки строки.

Условия задачи:
1) Используем pipeline('summarization')
2) max_length=15, min_length=5 — фиксированные значения
3) На вход подаём текст целиком
4) Результат печатаем в stdout (опционально сохраняем в файл --output)
"""

import argparse
import os
import sys
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def read_text_file(filepath: str) -> str:
    """
    Читает текст из файла (UTF-8) и проверяет, что он не пустой.
    Возвращает содержимое файла строкой.

    Исключения:
      - FileNotFoundError → если файл отсутствует
      - ValueError        → если файл пуст
      - IOError           → если любая другая ошибка чтения
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            raise ValueError("File is empty")
        return text
    except Exception as e:
        # Оборачиваем любые проблемы чтения в единый понятный текст
        raise IOError(f"Error reading file '{filepath}': {e}")


def write_text_file(filepath: str, text: str) -> None:
    """
    Записывает текст в файл (UTF-8), создавая директорию при необходимости.

    Исключение:
      - IOError → при ошибке записи
    """
    try:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        raise IOError(f"Error writing to file '{filepath}': {e}")


def create_summarizer(model_name: str = "facebook/bart-large-cnn",
                      device: Optional[int] = None):
    """
    Создаёт и возвращает summarization-pipeline.

    Почему грузим модель вручную:
      - Используем use_safetensors=True, чтобы НЕ обращаться к .bin,
        тем самым избегаем ограничений на torch.load (актуальные политики безопасности).

    Параметры:
      - model_name: имя модели на Hugging Face
      - device: 0 → CUDA (GPU), -1 → CPU, None → определить автоматически
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    try:
        # Токенайзер
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Модель (важно: safetensors)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            use_safetensors=True,   # критично: не трогаем .bin → безопасная загрузка
            torch_dtype="auto",
        )
    except Exception as e:
        # Явно сигналим, что проблема на этапе загрузки модели/токенайзера
        raise RuntimeError(f"Error loading model '{model_name}': {e}")

    # Собираем pipeline суммаризации
    return pipeline(
        "summarization",
        model=mdl,
        tokenizer=tok,
        device=device,
        truncation=True,  # аккуратно обрежет слишком длинный вход на уровне токенизации
    )


def summarize_text(summarizer, text: str) -> str:
    """
    Генерирует краткое резюме текста через pipeline('summarization').

    ВАЖНО: по условию фиксируем длины генерации токенов:
      max_length=15, min_length=5

    Возвращает резюме строкой (без пост-обрезки по словам).
    """
    try:
        out = summarizer(
            text,
            max_length=15,   # фиксированное ограничение токенов по ТЗ
            min_length=5,    # минимальная длина токенов по ТЗ
            do_sample=False, # детерминированный результат (beam search)
            num_beams=4,     # ширина перебора лучей
            early_stopping=True,
        )
        # pipeline возвращает список с одним словарём
        return out[0]["summary_text"].strip()
    except Exception as e:
        # Любая ошибка генерации → единый понятный текст ошибки
        raise RuntimeError(f"Error during summarization: {e}")


def main() -> None:
    """
    Точка входа CLI:
      --input  : путь к входному .txt
      --output : (опционально) путь для сохранения результата
      --model  : имя модели (по умолчанию BART)
      --cpu    : форсировать CPU даже при наличии CUDA
    """
    ap = argparse.ArgumentParser(description="Text summarization (prints result to stdout).")
    ap.add_argument("--input",  required=True, help="Path to input text file")
    ap.add_argument("--output", help="(Optional) Path to save summary file")
    ap.add_argument("--model",  default="facebook/bart-large-cnn", help="HF model name")
    ap.add_argument("--cpu",    action="store_true", help="Force CPU even if CUDA is available")
    args = ap.parse_args()

    # 1) Читаем вход
    try:
        text = read_text_file(args.input)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # 2) Создаём пайплайн
    try:
        device = -1 if args.cpu else (0 if torch.cuda.is_available() else -1)
        summarizer = create_summarizer(args.model, device=device)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(2)

    # 3) Генерируем резюме с фиксированными max_length/min_length
    try:
        summary = summarize_text(summarizer, text)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(3)

    # 4) Печатаем результат в stdout — требование задачи
    print(summary)

    # (Необязательный шаг) Сохраняем в файл, если передан --output
    if args.output:
        try:
            write_text_file(args.output, summary + "\n")
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(4)


if __name__ == "__main__":
    main()
