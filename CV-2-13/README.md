# Улучшенный OCR с предобработкой (CV-2-13)


---

## ✨ Возможности

- Линейное усиление контраста/яркости (`alpha`, `beta`).
- Бинаризация и режимы предобработки из `pic2txt.py`:
  `default | document | newspaper | inverted | comic`.
- Масштабирование изображения (по умолчанию **×2**).
- OCR: `pytesseract.image_to_string()` (движок **Tesseract OCR**).
- Сравнение метрик:
  - при `--truth` → **Accuracy, % (по словам)**;
  - иначе → **Average confidence (Tesseract)**.
- Сохранение артефактов: предобработанное изображение и превью «до/после».
- Отчёт в **JSON/CSV** с параметрами, метриками и сведениями об окружении.

---

## 📁 Структура проекта

```text
CV-2-13/
├─ cv213_runner.py                 # основной раннер: baseline → preprocess → OCR → метрики/отчёт
├─ pic2txt.py                      # модуль предобработки: бинаризация/режимы/масштаб + загрузка
├─ brightness_and_contrast.py      # модуль усиления контраста/яркости (alpha/beta)
├─ requirements.txt
└─ README.md
```

🔧 Требования
Python 3.9+

Установленный Tesseract OCR (проверьте: tesseract --version).
Если Tesseract не в PATH, укажите путь флагом --tesseract-cmd.

Python-библиотеки:
```
txt
Копировать код
# requirements.txt
numpy>=1.24
opencv-python>=4.8
Pillow>=10.0
pytesseract>=0.3.10
```
🧩 Установка
```
pip install -r requirements.txt
```
▶️ Быстрый старт
Минимальный запуск (выполняет все 5 пунктов задания и сохраняет артефакты):
```

python cv213_runner.py img.png --lang eng --scale 2 \
  --save pre.png --preview sbs.png --report rep.json
```
```
python cv213_runner.py img.png --lang eng --scale 2 \
  --truth truth.txt --save pre.png --preview sbs.png --report rep.json
Явный режим + параметры контраста/PSM/DPI:
```
```
bash
Копировать код
python cv213_runner.py img.png --lang eng+rus --psm 6 --dpi 300 \
  --mode document --scale 2 --alpha 1.3 --beta 20 \
  --save pre.png --preview sbs.png --report rep.csv --report-format csv
Если Tesseract не в PATH:
```
bash
Копировать код
python cv213_runner.py img.png --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
⚙️ Параметры CLI
Аргумент	Описание	По умолчанию
```
image	Путь к исходному изображению	—
--lang	Язык(и) Tesseract (напр. eng, eng+rus)	eng
--psm	Page Segmentation Mode (0..13)	6
--dpi	DPI-хинт для Tesseract	300
--mode	Режим предобработки из pic2txt.py	document
--scale	Масштаб увеличения	2.0
--alpha, --beta	Контраст/яркость (linear)	1.5, 30.0
--no-contrast	Отключить усиление контраста/яркости	выкл.
--truth	Путь к эталонному тексту для точной метрики	—
--save	PNG для сохранения предобработанного изображения	—
--preview	PNG для сохранения превью «до/после»	—
--report	Путь к отчёту (.json или .csv)	—
--report-format	Формат отчёта (json | csv)	json
--tesseract-cmd	Путь к tesseract.exe, если не в PATH	—
--log-level	DEBUG | INFO | WARNING | ERROR	INFO
```
📊 Метрики и цель ≥ 90%
```
С эталоном (--truth): считается Accuracy, % (по словам) на основе Левенштейна по токенам; печатается Improvement (pp) и статус достижения цели ≥ 90%.

Без эталона: сравнивается Average confidence из pytesseract.image_to_data.

Пример финального блока:
```
```

=== METRICS ===
Accuracy, % (by truth words) — baseline:    85.30
Accuracy, % (by truth words) — preprocessed: 92.45
Improvement (pp): 7.15
Target ≥90% words correct: OK: ≥ 90%
```
