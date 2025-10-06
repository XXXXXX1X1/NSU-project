#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cv213_runner.py — единый раннер для CV-2-13.

Назначение
----------
1) Делает baseline OCR без предобработки.
2) Делает предобработку: усиление контраста/яркости (из brightness_and_contrast.py),
   затем бинаризация/масштаб ×2 и режимы (из pic2txt.py), и повторяет OCR.
3) Сравнивает результаты (по эталону truth или по средней уверенности Tesseract),
   печатает улучшение (pp) и, по желанию, сохраняет:
   - предобработанное изображение (--save),
   - превью до/после (--preview),
   - отчёт (--report в JSON или CSV).

Зависимости
-----------
Python 3.9+, numpy, opencv-python, pillow, pytesseract, tesseract-ocr установлен в системе.
Файлы brightness_and_contrast.py и pic2txt.py должны лежать рядом.

Примеры
-------
  python cv213_runner.py img.png --lang eng --scale 2 --save pre.png --preview sbs.png --report rep.json
  python cv213_runner.py img.png --lang eng+rus --psm 6 --dpi 300 --truth truth.txt --report rep.csv
  python cv213_runner.py img.png --no-contrast --mode newspaper --alpha 1.2 --beta 10

Автор: выровнено под критерии качества (валидность, модульность, обработка ошибок, докстринги).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
from PIL import Image
import pytesseract

# Гарантируем импорт локальных модулей из текущей папки (кроссплатформенно)
_CURR_DIR = Path(__file__).parent.resolve()
if str(_CURR_DIR) not in sys.path:
    sys.path.insert(0, str(_CURR_DIR))

# Эти модули предоставлены пользователем и должны лежать рядом
try:
    import brightness_and_contrast as bc  # adjust_brightness_contrast(channel, alpha, beta)
except Exception as e:
    raise SystemExit(f"[ERROR] Не удалось импортировать brightness_and_contrast.py: {e}")

try:
    import pic2txt as p2t  # load_image(path) и preprocess(img_bgr, mode, scale)
except Exception as e:
    raise SystemExit(f"[ERROR] Не удалось импортировать pic2txt.py: {e}")

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

# ---------- Конфигурация и логирование ----------

@dataclass
class Config:
    image: Path
    lang: str = "eng"
    psm: int = 6
    dpi: int = 300
    alpha: float = 1.5
    beta: float = 30.0
    scale: float = 2.0
    mode: str = "document"  # default|document|newspaper|inverted|comic (из pic2txt.py)
    truth: Optional[Path] = None
    save: Optional[Path] = None
    preview: Optional[Path] = None
    report: Optional[Path] = None
    report_format: str = "json"  # json|csv
    tesseract_cmd: Optional[Path] = None
    no_contrast: bool = False
    log_level: str = "INFO"

def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(levelname)s: %(message)s"
    )

# ---------- Утилиты ----------

def tokenize_words(text: str) -> list[str]:
    """Простая токенизация по буквенно-цифровым блокам (RU/EN)."""
    return WORD_RE.findall((text or "").lower())

def levenshtein_distance(a: list[str], b: list[str]) -> int:
    """Векторно-эффективная реализация (O(n*m), память O(m))."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]

def word_accuracy(truth: str, pred: str) -> float:
    """
    Точность по словам: 1 - Levenshtein(truth_words, pred_words)/len(truth_words).
    Если truth пуст — 0.0.
    """
    tw, pw = tokenize_words(truth), tokenize_words(pred)
    if not tw:
        return 0.0
    dist = levenshtein_distance(tw, pw)
    return max(0.0, 1.0 - dist / max(1, len(tw)))

def avg_confidence(image_bgr: np.ndarray, lang: str, psm: Optional[int], dpi: Optional[int]) -> float:
    """Средняя уверенность из pytesseract.image_to_data; NaN, если недоступно."""
    cfg = []
    if psm is not None:
        cfg.append(f"--psm {psm}")
    if dpi is not None:
        cfg.append(f"--dpi {dpi}")
    data = pytesseract.image_to_data(
        image_bgr, lang=lang, config=" ".join(cfg) if cfg else None,
        output_type=pytesseract.Output.DICT
    )
    vals = [float(c) for c in data.get("conf", []) if c not in ("-1", -1, None, "")]
    return float(np.mean(vals)) if vals else float("nan")

def to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def make_sbs(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """Горизонтальное превью «до/после» с выравниванием по высоте (без осей)."""
    h = max(left_bgr.shape[0], right_bgr.shape[0])
    def pad(im):
        if im.shape[0] == h:
            return im
        t = (h - im.shape[0]) // 2
        b = h - im.shape[0] - t
        return cv2.copyMakeBorder(im, t, b, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return cv2.hconcat([pad(left_bgr), pad(right_bgr)])

def enhance_contrast_brightness(img_bgr: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Поканальное линейное усиление (B,G,R) из brightness_and_contrast.py."""
    out = np.empty_like(img_bgr)
    for c in range(3):
        out[:, :, c] = bc.adjust_brightness_contrast(img_bgr[:, :, c], alpha=alpha, beta=beta)
    return out

def gather_env_info() -> Dict[str, Any]:
    """Собирает информацию об окружении для отчёта."""
    try:
        tess_ver = str(pytesseract.get_tesseract_version())
    except Exception:
        tess_ver = "unknown"
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": Image.__version__,
        "pytesseract": getattr(pytesseract, "__version__", "unknown"),
        "tesseract": tess_ver,
    }

# ---------- OCR-поток ----------

def ocr_text(img_bgr: np.ndarray, lang: str, psm: int, dpi: int) -> Tuple[str, float]:
    """Возвращает (текст, средняя уверенность)."""
    cfg = f"--psm {psm} --dpi {dpi}"
    text = pytesseract.image_to_string(to_pil(img_bgr), lang=lang, config=cfg).strip()
    conf = avg_confidence(img_bgr, lang, psm, dpi)
    return text, conf

def preprocess_with_params(src_bgr: np.ndarray, mode: str, scale: float,
                           alpha: float, beta: float, no_contrast: bool) -> np.ndarray:
    """
    1) Контраст/яркость (если не отключено),
    2) Бинаризация/масштаб/режим — через p2t.preprocess (использует твой код).
    """
    adj = src_bgr if no_contrast else enhance_contrast_brightness(src_bgr, alpha=alpha, beta=beta)
    th = p2t.preprocess(adj, mode=mode, scale=scale)
    # На выход может прийти GRAY — приводим к BGR для единообразия интерфейса
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR) if th.ndim == 2 else th

# ---------- Валидация ввода ----------

def validate_args(cfg: Config) -> None:
    if not cfg.image.exists() or not cfg.image.is_file():
        raise SystemExit(f"[ERROR] image not found: {cfg.image}")
    if cfg.truth and (not cfg.truth.exists() or not cfg.truth.is_file()):
        raise SystemExit(f"[ERROR] truth file not found: {cfg.truth}")
    if cfg.psm < 0 or cfg.psm > 13:
        raise SystemExit("[ERROR] --psm должен быть в диапазоне 0..13")
    if cfg.scale <= 0:
        raise SystemExit("[ERROR] --scale должен быть > 0")
    if cfg.alpha <= 0:
        raise SystemExit("[ERROR] --alpha должен быть > 0")
    if cfg.report and cfg.report_format not in ("json", "csv"):
        raise SystemExit("[ERROR] --report-format должен быть json или csv")
    # Проверяем нечётность типичных параметров в p2t внутри самого p2t (наш код их не принимает),
    # поэтому здесь дополнительных проверок не требуется.

# ---------- Главная функция ----------

def run(cfg: Config) -> int:
    setup_logging(cfg.log_level)
    logging.debug("Конфигурация: %s", asdict(cfg))

    # Путь к tesseract, если задан
    if cfg.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = str(cfg.tesseract_cmd)

    # Загрузка исходника через p2t.load_image (используем код пользователя)
    try:
        src_bgr = p2t.load_image(str(cfg.image))
    except Exception as e:
        logging.error("Не удалось загрузить изображение через pic2txt.load_image: %s", e)
        return 2
    if src_bgr is None:
        logging.error("pic2txt.load_image вернул None для: %s", cfg.image)
        return 2

    # Baseline OCR
    baseline_text, baseline_conf = ocr_text(src_bgr, cfg.lang, cfg.psm, cfg.dpi)

    # Предобработка (контраст/яркость + режим/масштаб из pic2txt)
    try:
        pre_bgr = preprocess_with_params(
            src_bgr, mode=cfg.mode, scale=cfg.scale,
            alpha=cfg.alpha, beta=cfg.beta, no_contrast=cfg.no_contrast
        )
    except Exception as e:
        logging.error("Ошибка предобработки (p2t.preprocess): %s", e)
        return 2

    # OCR после предобработки
    pre_text, pre_conf = ocr_text(pre_bgr, cfg.lang, cfg.psm, cfg.dpi)

    # Метрики
    truth_text = None
    if cfg.truth:
        try:
            truth_text = cfg.truth.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logging.error("Не удалось прочитать truth: %s", e)
            return 2

    if truth_text:
        base_acc = word_accuracy(truth_text, baseline_text) * 100.0
        pre_acc = word_accuracy(truth_text, pre_text) * 100.0
        improvement_pp = pre_acc - base_acc
        label = "Accuracy, % (by truth words)"
    else:
        base_acc = baseline_conf
        pre_acc = pre_conf
        improvement_pp = pre_acc - base_acc
        label = "Avg confidence (Tesseract)"

    # Вывод (читабельный, без лишнего)
    print("\n=== BASELINE (no preprocessing) ===\n")
    print(baseline_text or "(пусто)")
    print(f"\n=== PREPROCESSED (mode={cfg.mode}, scale×{cfg.scale:.2f}, "
          f"{'no-contrast' if cfg.no_contrast else f'alpha={cfg.alpha}, beta={cfg.beta}'}) ===\n")
    print(pre_text or "(пусто)")

    print("\n=== METRICS ===")
    base_val = float(base_acc) if base_acc == base_acc else float("nan")
    pre_val = float(pre_acc) if pre_acc == pre_acc else float("nan")
    print(f"{label} — baseline:    {base_val:.2f}")
    print(f"{label} — preprocessed: {pre_val:.2f}")
    print(f"Improvement (pp): {improvement_pp:.2f}")
    if truth_text:
        print(f"Target ≥90% words correct: {'OK: ≥ 90%' if pre_acc >= 90.0 else 'NEEDS WORK: < 90%'}")

    # Сохранения
    try:
        if cfg.save:
            cfg.save.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cfg.save), pre_bgr)
            print(f"[OK] Предобработанное изображение сохранено: {cfg.save}")
        if cfg.preview:
            cfg.preview.parent.mkdir(parents=True, exist_ok=True)
            sbs = make_sbs(src_bgr, pre_bgr)
            cv2.imwrite(str(cfg.preview), sbs)
            print(f"[OK] Превью до/после сохранено: {cfg.preview}")
        if cfg.report:
            cfg.report.parent.mkdir(parents=True, exist_ok=True)
            env = gather_env_info()
            rep: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "image": str(cfg.image),
                "params": {
                    "lang": cfg.lang, "psm": cfg.psm, "dpi": cfg.dpi,
                    "alpha": None if cfg.no_contrast else cfg.alpha,
                    "beta": None if cfg.no_contrast else cfg.beta,
                    "mode": cfg.mode, "scale": cfg.scale,
                    "no_contrast": cfg.no_contrast
                },
                "metrics": {
                    "mode": "truth_word_accuracy" if truth_text else "tesseract_avg_confidence",
                    "baseline": None if np.isnan(base_val) else round(base_val, 4),
                    "preprocessed": None if np.isnan(pre_val) else round(pre_val, 4),
                    "improvement_pp": None if np.isnan(improvement_pp) else round(float(improvement_pp), 4),
                    "target_90_passed": bool(truth_text and pre_acc >= 90.0),
                },
                "texts": {
                    "baseline": baseline_text,
                    "preprocessed": pre_text
                },
                "env": env
            }
            if cfg.report_format == "json" or cfg.report.suffix.lower() == ".json":
                cfg.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            elif cfg.report_format == "csv" or cfg.report.suffix.lower() == ".csv":
                # Плоская строка для CSV
                flat = {
                    "timestamp": rep["timestamp"],
                    "image": rep["image"],
                    "lang": cfg.lang,
                    "psm": cfg.psm,
                    "dpi": cfg.dpi,
                    "alpha": "" if cfg.no_contrast else cfg.alpha,
                    "beta": "" if cfg.no_contrast else cfg.beta,
                    "mode": cfg.mode,
                    "scale": cfg.scale,
                    "no_contrast": cfg.no_contrast,
                    "metric_mode": rep["metrics"]["mode"],
                    "baseline": rep["metrics"]["baseline"],
                    "preprocessed": rep["metrics"]["preprocessed"],
                    "improvement_pp": rep["metrics"]["improvement_pp"],
                    "target_90_passed": rep["metrics"]["target_90_passed"],
                    "python": rep["env"]["python"],
                    "numpy": rep["env"]["numpy"],
                    "opencv": rep["env"]["opencv"],
                    "pillow": rep["env"]["pillow"],
                    "pytesseract": rep["env"]["pytesseract"],
                    "tesseract": rep["env"]["tesseract"],
                }
                with cfg.report.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
                    writer.writeheader()
                    writer.writerow(flat)
            else:
                raise SystemExit("[ERROR] Неподдерживаемый формат отчёта (используй json или csv).")
            print(f"[OK] Отчёт сохранён: {cfg.report}")
    except Exception as e:
        logging.error("Ошибка при сохранении результатов: %s", e)
        return 2

    return 0

# ---------- CLI ----------

def parse_args(argv: list[str]) -> Config:
    p = argparse.ArgumentParser(
        description="CV-2-13: baseline vs preprocessed (использует brightness_and_contrast.py и pic2txt.py)."
    )
    p.add_argument("image", type=Path, help="Путь к изображению")
    p.add_argument("--lang", default="eng", help="Язык(и) Tesseract, напр. eng или eng+rus")
    p.add_argument("--psm", type=int, default=6, help="Page Segmentation Mode (0..13), по умолчанию 6")
    p.add_argument("--dpi", type=int, default=300, help="DPI-хинт для Tesseract")
    p.add_argument("--alpha", type=float, default=1.5, help="Контраст (alpha>0), по умолчанию 1.5")
    p.add_argument("--beta", type=float, default=30.0, help="Яркость (beta), по умолчанию 30.0")
    p.add_argument("--scale", type=float, default=2.0, help="Масштаб увеличения (>0). По заданию ×2")
    p.add_argument("--mode", choices=["default", "document", "newspaper", "inverted", "comic"],
                   default="document", help="Режим предобработки из pic2txt.py")
    p.add_argument("--truth", type=Path, default=None, help="Путь к эталонному тексту (для метрики >90%)")
    p.add_argument("--save", type=Path, default=None, help="PNG для сохранения предобработанного изображения")
    p.add_argument("--preview", type=Path, default=None, help="PNG превью до/после")
    p.add_argument("--report", type=Path, default=None, help="Путь к отчёту (.json или .csv)")
    p.add_argument("--report-format", choices=["json", "csv"], default="json", help="Формат отчёта (по умолчанию json)")
    p.add_argument("--tesseract-cmd", type=Path, default=None, help="Путь к tesseract.exe, если не в PATH")
    p.add_argument("--no-contrast", action="store_true", help="Отключить усиление контраста/яркости (использовать только p2t.preprocess)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Уровень логирования")
    args = p.parse_args(argv)

    cfg = Config(
        image=args.image,
        lang=args.lang,
        psm=args.psm,
        dpi=args.dpi,
        alpha=args.alpha,
        beta=args.beta,
        scale=args.scale,
        mode=args.mode,
        truth=args.truth,
        save=args.save,
        preview=args.preview,
        report=args.report,
        report_format=args.report_format,
        tesseract_cmd=args.tesseract_cmd,
        no_contrast=args.no_contrast,
        log_level=args.log_level,
    )
    validate_args(cfg)
    return cfg

def main() -> None:
    cfg = parse_args(sys.argv[1:])
    rc = run(cfg)
    raise SystemExit(rc)

if __name__ == "__main__":
    main()
