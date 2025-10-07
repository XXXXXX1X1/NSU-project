# DA-3-15 — Feature Interaction (Комбинация категориальных признаков)

---

## ⚙️ Функционал

✅ Принимает данные из:
- встроенного датасета `iris` (`--source iris`)
- пользовательского CSV (`--csv path`)

✅ Преобразует числовые признаки в категории с помощью **биннинга**:
- квантильный (`q3`, `q4`, …)
- по заданным границам (`cut:0,10,20,100`)

✅ Создаёт новый признак:


✅ Считает частоты комбинаций и выводит **топ-N** (по умолчанию 5)

✅ Проверяет входные данные и даёт понятные сообщения об ошибках

---

## 🧩 Установка

```bash
# создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# установить зависимости
pip install pandas scikit-learn
```
## 🚀 Примеры запуска
1️⃣ Использовать встроенный датасет Iris
```
python feature_combo.py --source iris --cat1 target_name --cat2 "sepal width (cm)" --bin2 q3
```
2️⃣ Два числовых признака (оба бинируются)
```
python feature_combo.py --source iris --cat1 "sepal length (cm)" --bin1 q4 --cat2 "petal width (cm)"  --bin2 q3
```
3️⃣ CSV с категориальными колонками
```
python feature_combo.py --csv data.csv --cat1 city --cat2 gender --topk 10
```
4️⃣ CSV с явными границами
```
python feature_combo.py --csv data.csv --cat1 age   --bin1 cut:0,18,35,50,120 --cat2 income --bin2 q5
```
## 📦 Аргументы CLI
Аргумент	Обязательно	Описание
```
--source {iris}	✅	Источник данных: встроенный iris
--csv PATH	✅	Путь к CSV-файлу (альтернатива --source)
--cat1 NAME	✅	Имя первого признака
--cat2 NAME	✅	Имя второго признака
--bin1	❌	Биннинг для cat1 (none, qN, cut:...)
--bin2	❌	Биннинг для cat2 (none, qN, cut:...)
--topk	❌	Сколько топ-комбинаций выводить (по умолчанию 5)
--sep	❌	Разделитель при объединении (по умолчанию _)
```
## 📊 Пример вывода
```
Создан признак взаимодействия: 'target_name_sepal width (cm)' (dtype=category)

Топ комбинаций:
                 combo  count   freq
versicolor_(1.999,2.9]     34 22.67%
      setosa_(3.2,4.4]     33 22.00%
 virginica_(1.999,2.9]     21 14.00%
   virginica_(2.9,3.2]     21 14.00%
      setosa_(2.9,3.2]     15 10.00%
```
