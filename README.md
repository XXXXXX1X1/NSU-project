Summarize CLI

Короткое резюме текста с помощью transformers и pipeline('summarization').
Скрипт читает входной .txt, генерирует краткое резюме (по умолчанию — модель BART) и печатает результат в stdout. По желанию — сохраняет в файл.

Возможности

Использует pipeline('summarization')

Фиксированные длины генерации max_length=15, min_length=5 (токены)

Чтение входного текста из файла, вывод в консоль, опциональная запись в файл

Поддержка GPU (CUDA) и CPU

Безопасная загрузка весов через safetensors

Примечание: max_length/min_length — это токены, а не слова.
Установка

Рекомендуется работать в виртуальном окружении:

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

Вариант A (GPU, CUDA 12.8)

Установи PyTorch для твоей CUDA (у тебя CUDA 12.8):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


Установи остальные зависимости:

pip install -r requirements.txt

Вариант B (CPU)
pip install -r requirements-cpu.txt


Содержимое файлов зависимостей:

requirements.txt:

transformers>=4.42,<5.0
safetensors>=0.4.2
sentencepiece>=0.1.99


(PyTorch ставится отдельно, как выше, под твою CUDA.)

requirements-cpu.txt:

transformers>=4.42,<5.0
torch>=2.6
safetensors>=0.4.2
sentencepiece>=0.1.99


sentencepiece нужен для моделей типа mT5; для BART не обязателен, но не мешает.

Использование

Базовый запуск (печать результата в консоль):

python summarize.py --input input.txt


С сохранением в файл:

python summarize.py --input input.txt --output headline.txt


С выбором модели:

# По умолчанию: facebook/bart-large-cnn (EN тексты)
python summarize.py --input input.txt --model facebook/bart-large-cnn

# Пример для многоязычной модели (RU/EN и др.)
python summarize.py --input input.txt --model csebuetnlp/mT5_multilingual_XLSum


Форсировать CPU, даже если есть CUDA:

python summarize.py --input input.txt --cpu

Параметры CLI

--input — путь к входному .txt (обязательно)

--output — путь к файлу для сохранения результата (опционально)

--model — имя модели Hugging Face (по умолчанию facebook/bart-large-cnn)

--cpu — использовать CPU даже при наличии CUDA

Структура проекта (минимум)
.
├─ summarize.py          # основной скрипт (CLI)
├─ requirements.txt
├─ requirements-cpu.txt
├─ .gitignore
├─ input.txt             # пример входа (можно переименовать/заменить)
└─ headline.txt          # пример выхода (если сохраняете)

Как это работает (коротко)

summary = pipeline('summarization')(text, max_length=15, min_length=5, ...)
Скрипт загружает модель и токенайзер вручную с use_safetensors=True и затем создает пайплайн суммаризации.
