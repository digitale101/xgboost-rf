# XGBoost 3.2.0 + RF_GROUPS (gruppi di feature per livello)

Modifica: con la variabile d'ambiente `XGB_RF_GROUPS_BY_LEVEL=<file>` ogni
livello di profondita' dell'albero usa solo le feature del gruppo
corrispondente (riga 1 del file = radice, riga 2 = livello 1, riga 3 =
livello 2, poi si ricomincia). Senza la variabile: XGBoost standard.

Build: tab **Actions** -> `build-xgboost-rf` -> **Run workflow**. A fine
build il pacchetto `.whl` e' in fondo alla pagina della run, sotto
**Artifacts**.
