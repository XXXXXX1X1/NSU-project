import torch
from transformers import pipeline, AutoTokenizer

MODEL = "csebuetnlp/mT5_multilingual_XLSum"

# выбираем GPU, если он доступен
device = 0 if torch.cuda.is_available() else -1

# токенайзер без предупреждений
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, legacy=False)

summarizer = pipeline(
    "summarization",
    model=MODEL,
    tokenizer=tokenizer,
    device=device
)

with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

result = summarizer(
    text,
    max_new_tokens=15,
    min_length=5,
    do_sample=False,
    num_beams=8,
    no_repeat_ngram_size=3,
    length_penalty=1.2
)

headline = result[0]["summary_text"].strip()
print("Заголовок:", headline)
