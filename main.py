import torch
from transformers import pipeline, AutoTokenizer

model_id = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

device = 0 if torch.cuda.is_available() else -1

summarizer = pipeline(
    "summarization",
    model=model_id,
    tokenizer=tokenizer,
    device=device
)

# Читаем текст из файла
with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

# Генерация заголовка
result = summarizer(text, max_length=15, min_length=5, do_sample=False, num_beams=4)
headline = result[0]["summary_text"]

print("Заголовок:", headline)
