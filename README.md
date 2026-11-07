# Molecular Activity Prediction — MISIS x MEPhI DRM TM

## Что сдаём организаторам

- Эксперименты и отрицательные результаты: `EXPERIMENTS.md`, `RESEARCH_LOG_20261107.md`, сценарии `screen_*.py`, `confirm_*.py` и JSON-отчёты.
- Финальный пайплайн: `run_current_best.py`, базовая модель `run_best.py`/`vacv_zero_hurdle.py` и добавление проверенной публикации `vacv_cage_publication.py`; групповая проверка базовой модели: `validate_zero_v2.py`, отчёты `zero_v2_validation.json` и `zero_v2_confirmation.json`.
- Выгрузка и проверка открытых данных: `collect_chembl.py`, `collect_vacv_2022.py`, `build_new_dataset.py`, `build_submission_database.py` и ссылки на первичные измерения в исходных таблицах.
- Итоговая база: `data/curated/vacv_database.csv` с колонками `SMILES,IC50_uM,CC50_uM,SI,DOI`. В ней 56 открытых точных наблюдений из шести публикаций. `data/curated/vacv_database_validation.json` содержит контрольные суммы и проверку SI. Пересборка: `.venv/bin/python build_new_dataset.py` и `.venv/bin/python build_submission_database.py`.

Подробное происхождение 34 записей ChEMBL, включая ID и URL трёх измерений, находится в `data/curated/vacv_new_observations.csv`; для 10 bornyl-записей публикация, DOI, измерения и URL находятся в `artifacts/external/bornyl_amides_expanded_source.csv`. Итоговая база предназначена для защиты и проверки источников. Базовая модель использует пять новых bornyl-структур после исключения пересечений с train; текущий лучший выпуск дополнительно применяет опубликованную серию адамантановых амидов. Остальные записи не добавлены автоматически, поскольку их включение не улучшило групповую проверку.

Ещё 12 точных измерений адамантановых амидов получены из открытой [публикации 10.3390/v15010029](https://pmc.ncbi.nlm.nih.gov/articles/PMC9865139/) и сохранены в `cage_amide_primary.csv` вместе с двумя цензурированными строками, исключёнными из итоговой базы. В статье концентрация цитотоксичности названа TC50; в пятиколоночной базе она отображается в запрошенную колонку CC50, а исходное определение сохранено в файле источника. `vacv_cage_publication.py` воспроизводит отдельный экспериментальный кандидат поверх лучшего CSV: использует только SMILES для структурного сопоставления и не читает тестовые концентрации или названия. Исходные три совпадения источника с train имеют относительное расхождение SI менее 0,001%; это проверка происхождения, а не гарантия качества на скрытом тесте.

## Запуск на другом компьютере

Клонируйте этот приватный репозиторий с доступом GitHub. Установите Python 3.13 и зависимости, затем скачайте с вкладки Data соревнования три файла `train_v2.csv`, `test.csv`, `sample_submit.csv` в `data/kaggle/` (эта папка исключена из Git). Из корня репозитория запустите:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_current_best.py
```

На Windows используйте `.venv\\Scripts\\python.exe` вместо `.venv/bin/python`. Скрипт сначала обучает базовую модель и проверяет SHA256 `submission_v2_stereo_hurdle.csv`, затем применяет `vacv_cage_publication.py` и проверяет SHA256 итогового `submission_v2_cage_publication.csv`. Оба публичных источника включены в репозиторий. Для запуска не нужны локальные кэши или предобученные модели. Если Kaggle изменит файлы, проверка остановит запуск с указанием несовпадающего файла.

Текущий лучший подтверждённый результат команды `misis x mephi drm tm` — **819 116.04045**, `submission_v2_cage_publication.csv`, публичное место **6 из 41** при проверке 7 ноября 2026 в 15:03 МСК. Лидер `seveneleven` показывает 0.00000; оценка рассчитывается примерно на 30% теста. В train_v2 организатор заменил 81 старую метку 10000 на 0; теперь нулей 83. SHA256 входов и лучшего CSV — `best_confirmed.json`. Только SMILES; запрещённые концентрационные поля не используются.

```bash
.venv/bin/python vacv_zero_hurdle.py --train data/kaggle/train_v2.csv \
  --test data/kaggle/test.csv --sample data/kaggle/sample_submit.csv \
  --source artifacts/external/bornyl_amides_expanded_source.csv \
  --calibration raw --conditional source_stereo_only \
  --model artifacts/zero_v2_hurdle_nn.joblib \
  --output submission_v2_stereo_hurdle.csv --report zero_v2_release.json
```

На 15 групповых разбиениях MSE 2 850 702 против 4 267 751 у заново обученного леса; на пяти дополнительных 2 907 210 против 4 520 415. Внешний снимок включён в Git, точные копии всех train/validation-молекул удалены. Публичные альтернативы forest 1 645 802.93915 и atom-pair 901 434.30767 хуже лучшего. Отчёты исследования сохраняют и неудачные проверки. Новые разбиения повторно используют те же молекулы.

Для судей отдельно создан чистый локальный репозиторий `misis-x-mephi-drm-tm-molecular`: только README, сценарий чекпоинта, код, разрешённые открытые выгрузки, конфигурация и проверенные отчёты. Его сокращённый код побайтово воспроизводит оба подтверждённых CSV; полные20 folds повторены. История ниже оставлена для исследования и не является текущим состоянием лидерборда.

## Исторические результаты и эксперименты

7 ноября изменились публичные оценки17 прежних отправок; исходный baseline тогда стал лучшим со13812033.63170. Повторная загрузка данных выявила исправленную train_v2. Все regraded баллы сохранены в `kaggle_regraded_scores_20261107_0332.json`.

Исторический результат до изменения оценок: **3 128 256.42100**, `submission_bornyl_isotonic.csv`, fismay, место 12 при проверке 7 ноября. Это на 21.9% ниже прежнего Brier-варианта. В модели используются хиральные соседи с проверкой совместимости заданной стереохимии одного каркаса и проверенные опубликованные измерения из `10.1002/cmdc.202100771`; лес и классификатор сохранены, к Brier-калибровке добавлена изотоническая кривая с заранее выбранным весом 50%.

```bash
.venv/bin/python vacv_external_neighbors.py --train data/kaggle/train.csv \
  --test data/kaggle/test.csv --sample data/kaggle/sample_submit.csv \
  --source artifacts/external/bornyl_amides_expanded_source.csv \
  --base-model artifacts/brier_neighbors.joblib --model artifacts/bornyl_isotonic.joblib \
  --neighbor-recipe scaffold-stereo --isotonic-weight .5 \
  --output submission_bornyl_isotonic.csv --report bornyl_isotonic_release.json
.venv/bin/python validate_external_neighbors.py \
  --source artifacts/external/bornyl_amides_expanded_source.csv \
  --reference-source artifacts/external/bornyl_amides_source.csv \
  --neighbor-recipe scaffold-stereo --all-history
```

Для воспроизведения этого запуска нужен вручную сверенный снимок `artifacts/external/bornyl_amides_expanded_source.csv`, SHA256 `2ac50f4859e3d16a201eb97cc3bb4d17c7a973da21e1fa3d0aa183814d000f18`. Он включён в репозиторий. Перед обучением можно выполнить `.venv/bin/python verify_source_snapshot.py`; `run_best.py` проводит ту же проверку автоматически. Автоматического публичного сборщика для этой рукописи нет.

Десять первичных измерений сверены по авторской рукописи; для доступных соединений дополнительно проверены структуры и IC50-записи ChEMBL. Их локальный снимок и исходные данные соревнования не включены в Git; контрольные суммы и происхождение сохранены в отчётах. Совпадения внешних молекул с train удаляются; в CV дополнительно удаляются совпадения с validation. Обозначения некоторых соединений в экспериментальном разделе рукописи расходятся с таблицей и обсуждением; сопоставление проверено по структурам и значениям ChEMBL. Опубликованный SI и отношение опубликованных средних концентраций сохранены раздельно.

Проверка `bornyl_stereo_validation.json`: первые 15 folds 6 129 160 → 5 903 649 по сравнению с предыдущей source-моделью; новые пять folds 6 637 382 → 6 123 709. Полный повтор расчёта из SMILES — bornyl_stereo_recomputed.json. Выигрыш сконцентрирован на больших SI и не даёт гарантии для закрытой части test. `source_improvement_uncertainty.json` показывает широкий диапазон описательного bootstrap по молекулам.

Изотоническая калибровка обучена на внутренних трёх групповых OOF-прогнозах. Вес 50% выбран на первых 15 folds, после чего подтверждён на следующих 10 и на пяти новых разбиениях seed=48372: 6 123 709 → 5 922 485. Все разбиения повторно используют те же 320 строк и не являются новым независимым датасетом. Отчёты: `bornyl_isotonic_validation.json`, `bornyl_isotonic_recomputed.json`, `bornyl_isotonic_release.json`. Публичные результаты всех пяти новых отправок и отклонённые эксперименты записаны в `two_hour_research_report.json`.

Прогноз индекса селективности малых молекул против vaccinia virus (VACV) по SMILES.

Подробная постановка, результаты первоначального аудита и план: [PROJECT_BRIEF.md](PROJECT_BRIEF.md). Правила работы в репозитории: [AGENTS.md](AGENTS.md).

План выхода в топ-3 с учётом текущего лидерборда и правил: [TOP3_STRATEGY.md](TOP3_STRATEGY.md).

**Предыдущий baseline:** `submission_brier_neighbors.csv`, публичный MSE **4 006 684.58056** (Kaggle, 7 ноября 2026). Он использует Brier-калибровку и ансамбль структурных соседей; команда воспроизведения приведена в конце README. Предыдущий подтверждённый `submission_calibrated.csv` получил 4 052 042.53827. Предыдущий `submission_smiles_syntax.csv` получил 4 319 947.19472. Модель использует только SMILES; калибровка вероятности обучается на внутренних групповых разбиениях. Параметры закреплены в [best_confirmed.json](best_confirmed.json). Сначала создаётся базовый артефакт для нового варианта:

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

Проверка на новой серии разбиений: `.venv/bin/python validate_brier_fresh.py`. Публичный MSE этого предыдущего варианта — **4 006 684.58056**. Актуальный лучший файл указан в начале README.

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

Проверка экспериментального контекста всех точных IC50/EC50–CC50 пар: `.venv/bin/python audit_assay_context.py`. Из 293 перекрёстных пар только 47 имеют совпадающие аннотированные клетки и длительность. Они остаются кандидатами для проверки по публикациям; автоматически в обучение не входят. Остальные пары явно помечены причинами несовместимости или отсутствующими условиями. Выгрузка сохраняет исходные и стандартизованные измерения, DOI, ID и URL обеих записей; лицензия указана в assay_context_candidates.LICENSE.md.

Замороженный ChemBERTa и граф исходных SMILES проверены совместно в `graph_chemberta_validation.json`: 25 старых folds 6 586 204 → 6 501 136; пять новых групповых folds seed48372 7 007 701 → 6 749 133. Это проверки на тех же 320 строках, а не независимые новые данные. Публичный MSE `submission_graph_chemberta.csv` — **4 182 341.42466**, поэтому лучший Brier-вариант сохранён. ChemBERTa предобучен на 100M молекул ZINC20; источник и MIT-лицензия: https://huggingface.co/DeepChem/ChemBERTa-100M-MLM. Тестовые SMILES участвуют в построении графа без меток; это явно трансдуктивная модель.

Воспроизведение требует основной среды и отдельной среды энкодера:

```bash
.venv/bin/python export_chemberta_smiles.py --train data/kaggle/train.csv \
  --test data/kaggle/test.csv --output artifacts/chemberta/smiles.json
python3 -m venv .venv-chemberta
.venv-chemberta/bin/python -m pip install -r requirements-chemberta.txt
.venv-chemberta/bin/python extract_chemberta_embeddings.py \
  --input artifacts/chemberta/smiles.json --output artifacts/chemberta/embeddings.npz \
  --report chemberta_encoder_report.json
.venv/bin/python vacv_graph_chemberta.py --train data/kaggle/train.csv \
  --test data/kaggle/test.csv --sample data/kaggle/sample_submit.csv \
  --features artifacts/chemberta/embeddings.npz --encoder-report chemberta_encoder_report.json \
  --base-model artifacts/brier_neighbors.joblib --model artifacts/graph_chemberta.joblib \
  --output submission_graph_chemberta.csv --report graph_chemberta_release.json
```

Релиз проверяет точные колонки и порядок 597 строк, конечность прогнозов, побитовую неизменность при подмене запрещённых столбцов и удалении всех прочих признаков, а также прогноз после сохранения/загрузки модели. Веса энкодера закреплены по revision и SHA256.

Прямая проверка выбранного графового/ChemBERTa кандидата с повторным обучением: `.venv/bin/python validate_graph_chemberta.py --all-history`. Без флага выполняется только новая серия seed48372.

Дополнительный источник 2023 года (`10.1016/j.ejmech.2023.115601`) собирается `.venv/bin/python collect_cidofovir_selectivity.py`: 36 согласованных точных опубликованных SI, отдельно 18 Vero и 18 A549. В источнике токсичность измеряется за 72 часа, антивирусная активность — за 96 часов; оба протокола сохранены, их эквивалентность не утверждается. Числа SI опубликованы непосредственно, а не созданы произвольным объединением опытов. Полные исходные единицы, отношения, идентификаторы и URL сохранены.

Лицензированная первичная серия adamantane-amides собирается `.venv/bin/python collect_cage_amide_primary.py`: 14 соединений, 12 точных опубликованных измерений и 2 неактивные/цензурированные строки без обучающих меток. TC50 сохраняется под исходным названием и определением автора; SI = TC50/IC50. Лицензия CC BY 4.0 проверена на странице статьи. На текущем сравнении добавление серии к лучшему варианту не дало убедительного дополнительного выигрыша; новая отправка с ней не выпускалась.

Дополнение источников графом SMILES с фиксированным весом .25 получило публичный MSE **3 692 980.86025** и не заменило лучший source-neighbor вариант. Полный отчёт — bornyl_graph_validation.json. Стресс-проверка `.venv/bin/python validate_external_scaffolds.py --source artifacts/external/bornyl_amides_source.csv` показывает пользу источников для родственных семейств, но не доказала улучшение при исключении целых внешних каркасов: MSE 7 561 227 → 7 566 679. Это ограничение экстраполяции, а не оценка закрытого Kaggle test.

Стереохимическое правило задаётся в stereo_neighbors.py: если ахиральный каркас одинаков, но обе заданные конфигурации несовместимы при изоморфном сопоставлении, такая структура исключается из ближайших соседей. Неуказанная стереохимия не считается доказанным конфликтом. Это эвристика структуры, проверенная на grouped CV; она не означает общего доказательства биологического механизма.
