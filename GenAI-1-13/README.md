<h1 align="center">GenAI-1-13 · Headline Summarizer</h1>

<p align="center">
  Генерация краткого заголовка к тексту новости на базе Hugging Face <code>transformers</code>.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg" alt="pytorch"></a>
  <a href="https://huggingface.co/docs/transformers"><img src="https://img.shields.io/badge/transformers-4.44%2B-yellow.svg" alt="transformers"></a>
  <a href="#"><img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-444.svg" alt="platform"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-0aa.svg" alt="license"></a>
</p>

---

## ✨ Что это

Небольшая CLI-утилита, которая берёт **текст новости** из файла и возвращает **краткий заголовок**.  
Используется `pipeline('summarization')`. По умолчанию длина заголовка: `min_length=5`, `max_length=15` (токены).

---

## 🚀 Возможности

- Простая команда из консоли.
- Поддержка CPU / CUDA (`--device auto|cpu|cuda`).
- Настройка модели и длин генерации.
- Обработка ошибок (нет файла, пустой файл, неверные параметры).
- Можно использовать как **библиотеку** (импорт функций).

---

## 🗂 Структура

```text
.
├─ summarize.py          # CLI и функции
├─ requirements.txt
├─ input.txt             # пример входного текста
└─ headline.txt          # сюда будет записан заголовок
```
📦 Требования и установка

Python: 3.10+ (подойдёт и 3.11)
```
pip install -r requirements.txt
```

requirements.txt:
```
transformers>=4.44.0
torch>=2.6.0
sentencepiece>=0.1.99
safetensors>=0.4.3
accelerate>=1.0.0

```
▶️ Запуск

Базовый вариант:
```
python summarize.py input.txt headline.txt
```

🔧 Аргументы CLI
```
input_file — путь к входному .txt с текстом.

output_file — путь к .txt для заголовка.

--model — модель на HF (по умолчанию: csebuetnlp/mT5_multilingual_XLSum).

--max-length — максимум токенов в заголовке (дефолт: 15).

--min-length — минимум токенов (дефолт: 5).

--device — auto | cpu | cuda (дефолт: auto).
```
🧩 Пример входа/выхода

input.txt
```
В Москве открылась первая в России клиника, где диагностика проводится с помощью ИИ.
Система ускоряет анализ МРТ и КТ, сокращая время постановки диагноза почти в два раза.
Минздрав планирует масштабировать проект по стране в 2026 году.
```

Команда
```
python summarize.py input.txt headline.txt --device cpu
```

headline.txt (пример)
```
Минздрав запускает систему ИИ для ускоренной диагностики
