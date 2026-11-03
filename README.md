# Рекомендательные системы

Новиков Кирилл Сергеевич, Б05-307.

[Домашнее задание 1](hw1/experiments.ipynb) — сравнение пяти моделей на Amazon Video Games.

Для запуска нужен Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r hw1/requirements.txt
cd hw1
jupyter lab experiments.ipynb
```

В ноутбуке нужно выполнить все ячейки сверху вниз. Данные скачиваются автоматически в `hw1/data` и не входят в репозиторий. Для запуска без интерфейса из `hw1`:

```bash
jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=3600 experiments.ipynb
```
