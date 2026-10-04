# XGBoost 3.2.0 + RF_GROUPS (gruppi di feature per livello) — pacchetto xgboost_rf

Modifica: con la variabile d'ambiente `XGB_RF_GROUPS_BY_LEVEL=<file>` ogni
livello di profondita' dell'albero usa solo le feature del gruppo
corrispondente (riga 1 del file = radice, riga 2 = livello 1, riga 3 =
livello 2, poi si ricomincia). Senza la variabile: XGBoost standard.

Il pacchetto si chiama `xgboost_rf` (pip: `xgboost-rf`, libreria
`xgboost_rf.dll`) e si installa accanto all'XGBoost standard:

    pip install --no-deps xgboost_rf-3.2.0-py3-none-win_amd64.whl

    import xgboost as xgb        # standard (GPU), invariato
    import xgboost_rf as xgbrf   # modificato (solo CPU)

I modelli addestrati con xgboost_rf si trasferiscono nella classe standard
con `save_raw()` -> `load_model()`: predizioni identiche, export ONNX e
joblib invariati.

File:
- `rf_patch.py` modifica `src/common/random.h` (ColumnSampler)
- `rf_rename_wheel.py` rinomina il .whl in xgboost_rf
- `rf_selftest.py` verifica gruppi, convivenza con lo standard e
  trasferimento dei modelli
- `.github/workflows/build.yml` build su Windows (solo CPU)

Build: tab **Actions** -> `build-xgboost-rf` -> **Run workflow** (parte
anche da sola a ogni commit). A fine build il pacchetto `.whl` e' in fondo
alla pagina della run, sotto **Artifacts**.
