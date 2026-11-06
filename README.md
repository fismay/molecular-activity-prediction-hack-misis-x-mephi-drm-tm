# Molecular Activity Prediction — MISIS x MEPhI DRM TM

Прогноз индекса селективности малых молекул против vaccinia virus (VACV) по SMILES.

Подробная постановка, результаты первоначального аудита и план: [PROJECT_BRIEF.md](PROJECT_BRIEF.md). Правила работы в репозитории: [AGENTS.md](AGENTS.md).

Первый рабочий baseline запускается так (исходные CSV получить после вступления в [соревнование Kaggle](https://www.kaggle.com/competitions/molecular-activity-prediction-hack)):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python vacv_baseline.py \
  --train data/kaggle/train.csv --test data/kaggle/test.csv \
  --sample data/kaggle/sample_submit.csv \
  --output submission.csv --report baseline_report.json
```

Baseline использует только SMILES, проводит 5 групповых разбиений по каноническим структурам, сравнивает четыре модели по MSE и создаёт `submission.csv`. Исходные файлы Kaggle исключены из репозитория.
