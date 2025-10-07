#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DA-3-15: Создание признаков — комбинация категориальных
-------------------------------------------------------
Задача:
1) Взять два категориальных признака (или превратить числовые в категориальные через биннинг).
2) Создать новый признак-взаимодействие: cat1 + '_' + cat2.
3) Установить dtype=category.
4) Посчитать частоты комбинаций.
5) Вывести топ-5 комбинаций (или другое число через --topk).

Источник данных:
- Либо встроенный датасет iris (через --source iris),
- Либо CSV-файл (через --csv path).

Примеры запуска:
- Iris: целевой класс + бины по ширине чашелистика (3 квантили):
    python feature_combo.py --source iris --cat1 target_name --cat2 "sepal width (cm)" --bin2 q3

- CSV с явными категориями:
    python feature_combo.py --csv data.csv --cat1 city --cat2 gender

- CSV с числовыми колонками → биннинг:
    python feature_combo.py --csv data.csv --cat1 age --cat2 income --bin1 q4 --bin2 cut:0,30,50,100
"""

from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# Попробуем импортировать iris; если sklearn нет, дадим понятную ошибку при использовании --source iris
try:
    from sklearn.datasets import load_iris
except Exception:
    load_iris = None


# ====== Пользовательские ошибки для аккуратных сообщений ======

class FeatureError(ValueError):
    """Исключение для ошибок в данных/параметрах с понятным текстом для пользователя."""
    pass


def ensure_non_empty_df(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Проверяем, что DataFrame не пустой."""
    if df is None or df.empty:
        raise FeatureError(f"{name} пуст или не загружен.")


def check_columns_exist(df: pd.DataFrame, cols: List[str]) -> None:
    """Проверяем, что необходимые столбцы существуют в DataFrame."""
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise FeatureError(f"В DataFrame отсутствуют столбцы: {miss}. Доступные: {list(df.columns)}")


def is_categorical_like(s: pd.Series) -> bool:
    """
    Проверяем, что признак категориальный-подобный:
    - category dtype (современная проверка через isinstance),
    - object (строки),
    - булев.
    """
    from pandas.api.types import CategoricalDtype
    return (isinstance(s.dtype, CategoricalDtype) or
            pd.api.types.is_object_dtype(s) or
            pd.api.types.is_bool_dtype(s))


def is_numeric(s: pd.Series) -> bool:
    """Проверяем, что признак числовой (int/float)."""
    return pd.api.types.is_numeric_dtype(s)


# ====== Загрузка данных ======

def load_iris_df() -> pd.DataFrame:
    """
    Загружаем iris как DataFrame.
    Добавляем человекочитаемый столбец target_name с именами классов,
    сразу делаем его категориальным.
    """
    if load_iris is None:
        raise FeatureError("scikit-learn недоступен: не могу загрузить iris. Установите scikit-learn или используйте --csv.")
    data = load_iris(as_frame=True)
    df: pd.DataFrame = data.frame.copy()
    df["target_name"] = df["target"].map(dict(enumerate(data.target_names))).astype("category")
    return df


def load_csv_df(path: str) -> pd.DataFrame:
    """Загружаем CSV; даём понятную ошибку при сбое."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise FeatureError(f"Не удалось загрузить CSV '{path}': {e}")
    ensure_non_empty_df(df, f"CSV '{path}'")
    return df


# ====== Описание биннинга числовых признаков ======

@dataclass
class BinSpec:
    """
    Параметры биннинга:
    - kind='none'  -> без биннинга (ожидаем, что признак уже категориальный)
    - kind='q'     -> квантильный биннинг на q корзин (q>=2)
    - kind='cut'   -> биннинг по явным границам edges (список чисел, отсортирован)
    """
    kind: str  # 'none' | 'q' | 'cut'
    q: Optional[int] = None
    edges: Optional[List[float]] = None

    @staticmethod
    def parse(spec: Optional[str]) -> "BinSpec":
        """
        Поддерживаемые форматы:
          - None / "" / "none"
          - "q3" / "q4" / "q5"
          - "cut:0,10,20,100"
        """
        if spec is None or spec.strip() == "" or spec.strip().lower() == "none":
            return BinSpec(kind="none")

        s = spec.strip().lower()
        if s.startswith("q"):
            # квантильный биннинг
            try:
                q = int(s[1:])
                if q < 2:
                    raise ValueError
            except Exception:
                raise FeatureError(f"Некорректный бин-спек '{spec}'. Ожидалось qN, где N>=2 (например, q3).")
            return BinSpec(kind="q", q=q)

        if s.startswith("cut:"):
            # явные границы
            tail = s.split(":", 1)[1].strip()
            try:
                edges = [float(x) for x in tail.split(",")]
            except Exception:
                raise FeatureError(f"Некорректный список границ в '{spec}'. Пример: cut:0,30,50,100")
            if len(edges) < 2 or sorted(edges) != edges:
                raise FeatureError(f"Границы должны быть отсортированы и содержать >=2 значений: '{spec}'")
            return BinSpec(kind="cut", edges=edges)

        raise FeatureError(f"Неизвестный формат биннинга: '{spec}'")


def bin_numeric_series(s: pd.Series, spec: BinSpec, colname: str) -> pd.Series:
    """
    Превращаем числовую колонку в категориальную по правилам BinSpec:
    - qcut: равное количество объектов в каждом бине (по квантилям), duplicates='drop' спасает при дублях.
    - cut: биннинг по явным границам (левая граница включена, правая открыта).
    """
    if spec.kind == "none":
        raise FeatureError(
            f"Столбец '{colname}' числовой, а биннинг не задан. Укажите --bin1/--bin2 (например, q3)."
        )

    if spec.kind == "q":
        # Квантильный биннинг
        try:
            cat = pd.qcut(s, q=spec.q, duplicates="drop")
        except Exception as e:
            raise FeatureError(f"Не удалось выполнить qcut для '{colname}' (q={spec.q}): {e}")
        # Превращаем интервалы в компактный строковый вид и далее в category
        cat = cat.astype(str).str.replace(r"\s+", "", regex=True).astype("category")
        return cat

    if spec.kind == "cut":
        # Явные границы
        try:
            cat = pd.cut(s, bins=spec.edges, include_lowest=True, right=False)
        except Exception as e:
            raise FeatureError(f"Не удалось выполнить cut для '{colname}' (edges={spec.edges}): {e}")
        cat = cat.astype(str).str.replace(r"\s+", "", regex=True).astype("category")
        return cat

    raise FeatureError(f"Неизвестный тип биннинга: {spec.kind}")


def ensure_categorical(df: pd.DataFrame, col: str, binspec: BinSpec) -> pd.Series:
    """
    Гарантируем, что возвращаем категориальный столбец:
    - если уже категориальный/объект/булев — нормализуем строки и делаем category;
    - если числовой — применяем биннинг (qcut/cut) и делаем category.
    """
    s = df[col]
    if is_categorical_like(s):
        # Убираем лишние пробелы, пустые строки заменяем на 'NA'
        return s.astype(str).str.strip().replace("", "NA").astype("category")
    if is_numeric(s):
        return bin_numeric_series(s, binspec, col)
    # На всякий случай: всё прочее приводим к строкам и делаем category
    return s.astype(str).str.strip().replace("", "NA").astype("category")


# ====== Основная логика задачи ======

def make_interaction(cat1: pd.Series, cat2: pd.Series, sep: str = "_") -> pd.Series:
    """
    Создаём признак-взаимодействие из двух категориальных серий:
    Просто конкатенируем значения строкой: cat1 + sep + cat2.
    """
    if len(cat1) != len(cat2):
        raise FeatureError("Длины столбцов для взаимодействия не совпадают.")
    # Защита от пустых/NaN: заменяем на 'NA'
    c1 = cat1.astype(str).replace({"": "NA", "nan": "NA", "None": "NA"})
    c2 = cat2.astype(str).replace({"": "NA", "nan": "NA", "None": "NA"})
    inter = (c1 + sep + c2).astype("category")
    return inter


def top_k_frequencies(s: pd.Series, k: int = 5) -> pd.DataFrame:
    """
    Считаем частоты комбинаций и возвращаем топ-k строк.
    Столбцы: combo (значение), count (кол-во), freq (доля [0..1]).
    """
    vc = s.value_counts(dropna=False)
    total = int(vc.sum())
    out = (
        vc.head(k)
        .to_frame(name="count")
        .assign(freq=lambda d: d["count"] / total)
        .reset_index(names="combo")
    )
    return out


def run_pipeline(
    df: pd.DataFrame,
    cat1_name: str,
    cat2_name: str,
    bin1_spec: Optional[str],
    bin2_spec: Optional[str],
    interaction_sep: str = "_",
    top_k: int = 5,
) -> Tuple[pd.DataFrame, str]:
    """
    Полный конвейер:
    1) Валидация данных и колонок.
    2) Приведение обоих признаков к категориальным (с биннингом при необходимости).
    3) Создание взаимодействия.
    4) Подсчёт топ-k частот.
    Возвращает (таблица_топ_k, имя_нового_признака).
    """
    ensure_non_empty_df(df, "Входной DataFrame")
    check_columns_exist(df, [cat1_name, cat2_name])

    # Парсим спецификации биннинга для каждого столбца
    b1 = BinSpec.parse(bin1_spec)
    b2 = BinSpec.parse(bin2_spec)

    # Гарантируем категориальные столбцы на входе к взаимодействию
    c1 = ensure_categorical(df, cat1_name, b1)
    c2 = ensure_categorical(df, cat2_name, b2)

    # Создаём признак-взаимодействие
    inter_name = f"{cat1_name}{interaction_sep}{cat2_name}"
    df[inter_name] = make_interaction(c1, c2, sep=interaction_sep)

    # Считаем топ-k комбинаций
    top_table = top_k_frequencies(df[inter_name], k=top_k)
    return top_table, inter_name


# ====== CLI (интерфейс командной строки) ======

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Определяем аргументы CLI и парсим их из argv."""
    p = argparse.ArgumentParser(
        description="DA-3-15: Создание взаимодействия двух категориальных признаков и вывод топ-5 комбинаций."
    )
    # Источник данных: либо iris, либо CSV (взаимоисключающие аргументы)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", choices=["iris"], help="Источник данных: iris (sklearn)")
    src.add_argument("--csv", help="Путь к CSV-файлу")

    # Имена колонок для взаимодействия (обязательные)
    p.add_argument("--cat1", required=True, help="Имя первого столбца (или 'target'/'target_name' для iris).")
    p.add_argument("--cat2", required=True, help="Имя второго столбца.")

    # Опциональный биннинг для числовых столбцов
    p.add_argument("--bin1", default=None,
                   help="Биннинг для cat1, если он числовой. Форматы: none | q3 | q4 | cut:0,10,20")
    p.add_argument("--bin2", default=None,
                   help="Биннинг для cat2, если он числовой. Форматы: none | q3 | q4 | cut:0,10,20")

    # Параметры вывода
    p.add_argument("--topk", type=int, default=5, help="Сколько топ-комбинаций выводить (по умолчанию 5).")
    p.add_argument("--sep", default="_", help="Разделитель для взаимодействия (по умолчанию '_').")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа CLI: сбор параметров, загрузка, конвейер, печать результата и код возврата."""
    try:
        args = parse_args(argv)

        # 1) Загружаем данные из выбранного источника
        if args.source == "iris":
            df = load_iris_df()
        else:
            df = load_csv_df(args.csv)

        # 2) Запускаем конвейер создания взаимодействия и подсчёта топ-комбинаций
        top_table, inter_name = run_pipeline(
            df=df,
            cat1_name=args.cat1,
            cat2_name=args.cat2,
            bin1_spec=args.bin1,
            bin2_spec=args.bin2,
            interaction_sep=args.sep,
            top_k=args.topk,
        )

        # 3) Печатаем результат в консоль
        print(f"\nСоздан признак взаимодействия: '{inter_name}' (dtype=category)")
        print("\nТоп комбинаций:")
        if not top_table.empty:
            # Красивый вывод: доли в процентах с двумя знаками
            top_table_fmt = top_table.copy()
            top_table_fmt["freq"] = (top_table_fmt["freq"] * 100.0).map(lambda x: f"{x:.2f}%")
            print(top_table_fmt.to_string(index=False))
        else:
            print("Нет данных для отображения.")

        return 0

    # Ниже аккуратно обрабатываем ошибки с понятными кодами выхода
    except FeatureError as fe:
        print(f"[ОШИБКА ДАННЫХ] {fe}", file=sys.stderr)
        return 2
    except FileNotFoundError as fnf:
        print(f"[ОШИБКА ФАЙЛА] {fnf}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nОперация прервана пользователем.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"[НЕОЖИДАННАЯ ОШИБКА] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Выходим с кодом, который сообщает об успехе/ошибке оболочке (удобно для CI/CD)
    raise SystemExit(main())
