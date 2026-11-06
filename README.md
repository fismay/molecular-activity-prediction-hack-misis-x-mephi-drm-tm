# Molecular Activity Prediction — MISIS x MEPhI DRM TM

Прогноз индекса селективности малых молекул против vaccinia virus (VACV) по SMILES.

Подробная постановка, результаты первоначального аудита и план: [PROJECT_BRIEF.md](PROJECT_BRIEF.md). Правила работы в репозитории: [AGENTS.md](AGENTS.md).

План выхода в топ-3 с учётом текущего лидерборда и правил: [TOP3_STRATEGY.md](TOP3_STRATEGY.md).

**Лучший подтверждённый вариант:** `submission_brier_neighbors.csv`, публичный MSE **4 006 684.58056** (Kaggle, 7 ноября 2026). Он использует Brier-калибровку и ансамбль структурных соседей; команда воспроизведения приведена в конце README. Предыдущий подтверждённый `submission_calibrated.csv` получил 4 052 042.53827. Предыдущий `submission_smiles_syntax.csv` получил 4 319 947.19472. Модель использует только SMILES; калибровка вероятности обучается на внутренних групповых разбиениях. Параметры закреплены в [best_confirmed.json](best_confirmed.json). Сначала создаётся базовый артефакт для нового варианта:

```bash
.venv/bin/python vacv_text_mixture.py \
  --train data/kaggle/train.csv --test data/kaggle/test.csv \
  --sample data/kaggle/sample_submit.csv --syntax-ensemble \
  --syntax-max-features 0.7 --conditional-transform raw \
  --calibrate-probability --output submission_calibrated.csv \
  --report calibrated_report.json --model artifacts/calibrated.joblib
```

На 15 групповых разбиениях MSE — 6 887 822 против 7 025 214 без калибровки. Публичный балл получен только на части теста и не гарантирует итоговый результат.

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

Первый проверенный внешний источник ChEMBL выгружается командой:

```bash
.venv/bin/python collect_chembl.py --output chembl_vacv_curated.csv --report chembl_validation.json
```

Скрипт сохраняет DOI, прямые ссылки на источник, ID исходных измерений, IC50/CC50 в мкМ, опубликованный SI и результат проверки отношения CC50/IC50. Записи с расхождением отмечаются `mismatch_review` и требуют ручной проверки перед обучением.

Дополнительные опубликованные SI для VACV (кандидаты для валидации) выгружаются ` .venv/bin/python discover_si.py `. Результат `chembl_si_candidates.csv` содержит ссылки на исходные измерения и DOI. Это отдельный список кандидатов: IC50 и CC50 для этих строк пока не соединены и не должны считаться проверенной полной базой.

Эксперимент с внешними опубликованными SI и смесью двух моделей:

```bash
.venv/bin/python vacv_blend.py \
  --train data/kaggle/train.csv --test data/kaggle/test.csv \
  --sample data/kaggle/sample_submit.csv --external chembl_si_candidates.csv \
  --output submission_blend.csv --report blend_report.json
```

В каждом валидационном разбиении исключаются внешние записи для молекул из локального train и validation. Конфликтующие повторы и внешние SI выше 10000 исключаются согласно заранее проведённой локальной проверке; фильтр и веса записаны в коде и отчёте. Ни одна из запрещённых колонок Kaggle не используется.

Для 29 тестовых строк с точным совпадением канонической структуры в train итоговый прогноз берётся из наблюдаемого SI, только если повторные метки этой структуры согласованы. Это допустимый структурный поиск по SMILES; его эффект на закрытом тесте не известен до отправки на Kaggle. Последний локальный grouped-CV MSE для остальных решений смеси: 9 791 590. `submission_blend.csv` проверяется на 597 строк, порядок `index`, конечность и названия колонок.

Текущий публичный балл смеси на Kaggle (6 ноября) — 11 506 193, хуже первого baseline 10 720 485. Следующий кандидат использует случайный лес с `min_samples_leaf=1`, `max_features=0.3` и только признаки из SMILES. Локально он превосходит исходный лес на трёх сериях групповых разбиений; публичный результат пока не измерен:

```bash
.venv/bin/python vacv_rf_tuned.py \
  --train data/kaggle/train.csv --test data/kaggle/test.csv \
  --sample data/kaggle/sample_submit.csv
```

Команда создаёт `submission_rf_tuned.csv` и `rf_tuned_report.json`. Подстановка исходных SI для совпавших train/test структур в этом кандидате отключена, так как предыдущая отправка с такой подстановкой ухудшила публичный результат.

Обновление: `submission_rf_tuned.csv` получил публичный MSE **10 317 731.51657**. Следующий кандидат оценивает вероятность режима SI=10000 по SMILES и условный SI остальных молекул. Это статистическая модель меток; происхождение самого режима не установлено.

```bash
.venv/bin/python vacv_rf_tuned.py \
  --train data/kaggle/train.csv --test data/kaggle/test.csv \
  --sample data/kaggle/sample_submit.csv --strategy mixture \
  --output submission_mixture.csv --report mixture_report.json
```

Его MSE на 15 групповых разбиениях — 13 206 603 против 13 581 411 у настроенного леса, выигрыш сохраняется на каждой из трёх серий seed. Результат Kaggle ещё не измерен. Журнал сравнений: [EXPERIMENTS.md](EXPERIMENTS.md).

Второй проверяемый источник с совместимыми опытами Vero/VACV Copenhagen, 4 дня, neutral red uptake:

```bash
.venv/bin/python collect_chembl.py --study recent \
  --output chembl_recent_curated.csv --report chembl_recent_validation.json
```

Самое сильное локальное улучшение — сохранение исходной текстовой записи SMILES для классификации режима SI=10000. Канонизация перед символьным TF-IDF удаляла значимую для этого датасета информацию. В этом варианте классификатор использует символьные фрагменты длиной 3–7, а обычный SI предсказывается случайным лесом на структурных признаках:

```bash
.venv/bin/python vacv_text_mixture.py \
  --train data/kaggle/train.csv --test data/kaggle/test.csv \
  --sample data/kaggle/sample_submit.csv
```

Выходы: `submission_smiles_text.csv`, `smiles_text_report.json`, `artifacts/smiles_text.joblib`. Средний MSE на тех же 15 разбиениях — **7 765 100**, улучшение на 41% относительно структурной смеси. Публичная оценка пока неизвестна. Модель принимает только SMILES; проверены прогноз из сохранённого артефакта, формат CSV, удаление всех остальных входных колонок и инвариантность при подмене запрещённых измерений. Версии использованного окружения закреплены в requirements.txt, хеши файлов данных записаны в отчёте.

Ограничение: модель может использовать особенности записи SMILES, характерные для источников данных, поэтому эквивалентная перезапись молекулы способна изменить прогноз. Это допустимый текстовый вход для данного pipeline, но результат не доказывает переносимость на новые источники или истинную биологическую активность.

Публичный балл `submission_smiles_text.csv` — **4 972 122.51526**, улучшение относительно настроенного леса более чем вдвое. Следующий кандидат объединяет текстовый классификатор с ExtraTrees на 190 признаках исходной записи SMILES: частоты символов, номера циклов, записи стереоцентров и сравнения с альтернативными сериализациями RDKit. Веса двух классификаторов фиксированы как 0.5/0.5; условная регрессия сохранена:

```bash
.venv/bin/python vacv_text_mixture.py \
  --train data/kaggle/train.csv --test data/kaggle/test.csv \
  --sample data/kaggle/sample_submit.csv --syntax-ensemble \
  --output submission_smiles_syntax.csv --report smiles_syntax_report.json \
  --model artifacts/smiles_syntax.joblib
```

MSE по тем же 15 разбиениям — **7 025 214**, против **7 765 100** у текстовой модели. Публичный балл ещё неизвестен. Внешние базы, названия соединений и экспериментальные метаданные не используются. Для воспроизведения необходимо сохранить исходные строки SMILES, так как эти признаки зависят от способа записи.

Обновление: `submission_smiles_syntax.csv` получил **4 319 947.19472** на публичном Kaggle и поднял команду на 5-е место из 26 на момент проверки. Следующий кандидат обучает условный SI в корневой шкале и возвращает прогноз в исходную шкалу перед смешиванием; `max_features` синтаксического ExtraTrees снижен до 0.3:

```bash
.venv/bin/python vacv_text_mixture.py \
  --train data/kaggle/train.csv --test data/kaggle/test.csv \
  --sample data/kaggle/sample_submit.csv --syntax-ensemble \
  --syntax-max-features 0.3 --conditional-transform sqrt \
  --output submission_smiles_robust.csv --report smiles_robust_report.json \
  --model artifacts/smiles_robust.joblib
```

Локальная оценка по 15 разбиениям — **6 914 428**, улучшение 1.6% относительно предыдущего кандидата; на seed=42 — 2 634 371 против 2 879 543. Значения исходного SI не изменяются и не обрезаются, MSE считается после обратного преобразования. Публичный результат этого варианта пока неизвестен.

Расширенный сборщик `.venv/bin/python collect_pairs.py` загружает IC50 из VACV-опытов и CC50 из связанных публикаций. Из 15 087 записей сформировано 78 однозначных пар по публикации и молекуле с DOI и ссылками на каждое измерение. Это **кандидаты для проверки экспериментального контекста**: в обучении они автоматически не используются. В списке встречаются несовместимые клеточные линии, биохимические IC50 и несовпадающая длительность опытов.

Кандидат с условной регрессией по трём ближайшим структурам (локальный MSE 6 772 771 против 6 887 822 у калиброванного варианта):

```bash
.venv/bin/python vacv_knn_blend.py --train data/kaggle/train.csv \
  --test data/kaggle/test.csv --sample data/kaggle/sample_submit.csv \
  --model artifacts/calibrated.joblib --output submission_knn_blend.csv
```

Это экспериментальный файл; его публичный балл ещё не измерен.

Новый ансамбль трёх структурных представлений запускается той же командой с флагом `--ensemble --output submission_neighbor_ensemble.csv`. Проверка: `.venv/bin/python validate_neighbor_models.py`; полный отчёт — `neighbor_validation_report.json`. Публичная оценка этого кандидата пока неизвестна.

Последний кандидат добавляет Brier-калибровку к ансамблю соседей:

```bash
.venv/bin/python vacv_brier_neighbors.py --train data/kaggle/train.csv \
  --test data/kaggle/test.csv --sample data/kaggle/sample_submit.csv \
  --base-model artifacts/calibrated.joblib --model artifacts/brier_neighbors.joblib \
  --output submission_brier_neighbors.csv
```

Проверка на новой серии разбиений: `.venv/bin/python validate_brier_fresh.py`. Публичный MSE этого варианта — **4 006 684.58056**, лучший подтверждённый результат.

Экспериментальная совместная калибровка конечного MSE:

```bash
.venv/bin/python vacv_brier_neighbors.py --train data/kaggle/train.csv \
  --test data/kaggle/test.csv --sample data/kaggle/sample_submit.csv \
  --base-model artifacts/calibrated.joblib --model artifacts/final_mse_neighbors.joblib \
  --final-mse --output submission_final_mse.csv
```

Публичный MSE — **4 332 821.85908**, хуже Brier-варианта. Проверка новой серии: `.venv/bin/python validate_end_to_end_fresh.py`.

Опциональная проверка TabICL (результаты оказались хуже final-MSE кандидата) воспроизводится в отдельной среде:

```bash
.venv/bin/python export_tabicl_features.py --train data/kaggle/train.csv
python3 -m venv .venv-tabicl
.venv-tabicl/bin/python -m pip install -r requirements-tabicl.txt
.venv-tabicl/bin/python validate_tabicl_classifier.py
.venv-tabicl/bin/python screen_tabicl_regressor.py
```

Для этих проверок нужны кэши validate_neighbor_models.py, validate_probability_stacking.py и validate_end_to_end_calibration.py. Предобученные веса скачиваются из официального TabICL; данные соревнования обрабатываются локально. Исходный проект: https://github.com/soda-inria/tabicl.

Кандидаты внешних пар CC50/EC50 собираются collect_ec50_pairs.py. Они требуют проверки совпадения клеток, штаммов и длительности и не входят в текущую модель.

Проверенная осторожная смесь (75% Brier и 25% final-MSE) создаётся так:

```bash
.venv/bin/python blend_submissions.py --best submission_brier_neighbors.csv \
  --candidate submission_final_mse.csv --sample data/kaggle/sample_submit.csv \
  --weight .25 --output submission_conservative_blend.csv \
  --report conservative_release_checks.json
```

Её публичный MSE **4 072 147.65629**, поэтому она не заменяет лучший Brier-вариант. Новая групповая проверка: `.venv/bin/python validate_conservative_fresh.py`.
