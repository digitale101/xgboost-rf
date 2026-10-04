"""
rf_selftest.py - verifica che xgboost_rf (XGBoost con la modifica [RF_GROUPS],
installato come pacchetto a se' stante) rispetti i gruppi per livello e
conviva con l'XGBoost standard nello stesso processo.

Addestra un piccolo modello su dati casuali con 3 gruppi (feature 0-2, 3-5,
6-8) e controlla, nodo per nodo, che a profondita' d sia usata solo una
feature del gruppo d mod 3. Esce con errore se qualcosa non torna.

[RF_ENV] Prove su xgboost_rf, per distinguere le cause in caso di errore:
  A) variabile impostata DOPO l'import, nello stesso processo: e' il caso
     d'uso reale del trainer (gruppi diversi da un train all'altro).
  B) variabile impostata PRIMA dell'avvio, in un processo figlio: se B passa
     e A no, la dll legge solo l'ambiente presente al caricamento; se
     falliscono entrambe, il filtro non agisce sugli split.
  C) senza variabile: la regola NON deve valere (la prova discrimina).
[RF_RENAME] Prove di convivenza con l'XGBoost standard (pacchetto "xgboost"):
  L) importare xgboost_rf NON deve caricare il pacchetto "xgboost" (nessun
     import assoluto rimasto nel pacchetto rinominato).
  D) stesso processo, variabile impostata: xgboost standard la ignora (split
     NON conformi > 0) e le due librerie native sono file diversi.
  E) passaggio del modello: un XGBRegressor addestrato con xgboost_rf,
     trasferito nella classe standard (save_raw -> load_model), deve dare
     predizioni identiche. E' il percorso del trainer verso joblib e ONNX.
Uso interno: "python rf_selftest.py --child <file gruppi>" esegue solo B.
"""
import os, sys, subprocess, tempfile

RF_MOD = "xgboost_rf"
GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
PARAMS = {"max_depth": 3, "tree_method": "hist", "eta": 0.3, "seed": 1}


def make_data():
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 9))
    y = X[:, 0] * X[:, 4] + X[:, 8] + 0.5 * X[:, 7] * X[:, 2] + rng.normal(size=3000) * 0.1
    return X, y


def violations(bst):
    df = bst.trees_to_dataframe()
    bad = ok = 0
    for _, t in df.groupby("Tree"):
        t = t.set_index("ID")
        root = t.index[t["Node"] == 0][0]
        stack = [(root, 0)]
        while stack:
            nid, d = stack.pop()
            row = t.loc[nid]
            if row["Feature"] == "Leaf":
                continue
            f = int(str(row["Feature"]).lstrip("f"))
            if f in GROUPS[d % len(GROUPS)]:
                ok += 1
            else:
                bad += 1
            stack.append((row["Yes"], d + 1))
            stack.append((row["No"], d + 1))
    return ok, bad


def train_and_count(mod):
    X, y = make_data()
    dm = mod.DMatrix(X, label=y)
    return violations(mod.train(PARAMS, dm, 60))


def lib_file(mod):
    lib = getattr(getattr(mod, "core", None), "_LIB", None)
    return str(getattr(lib, "_name", "?"))


# ---- processo figlio: prova B (variabile gia' presente all'avvio) ----------
if len(sys.argv) >= 3 and sys.argv[1] == "--child":
    import importlib
    rf = importlib.import_module(RF_MOD)
    ok, bad = train_and_count(rf)
    print(f"{ok} {bad}")
    sys.exit(0)

path = os.path.join(tempfile.gettempdir(), "rf_groups_selftest.txt")
with open(path, "w") as fh:
    fh.write("\n".join(",".join(map(str, g)) for g in GROUPS))

import importlib  # noqa: E402
rf = importlib.import_module(RF_MOD)  # PRIMA di impostare la variabile: prova A

# L) nessun caricamento implicito del pacchetto standard
leak = "xgboost" in sys.modules
print(f"L import di {RF_MOD} carica anche 'xgboost': {'SI' if leak else 'no'}")

# A) variabile impostata dopo l'import
os.environ["XGB_RF_GROUPS_BY_LEVEL"] = path
okA, badA = train_and_count(rf)
print(f"A con gruppi (impostati dopo import): split conformi {okA}, NON conformi {badA}")

# C) senza variabile
os.environ.pop("XGB_RF_GROUPS_BY_LEVEL")
okC, badC = train_and_count(rf)
print(f"C senza gruppi:                       split conformi {okC}, NON conformi {badC}")

# B) variabile presente all'avvio, processo figlio
env = dict(os.environ)
env["XGB_RF_GROUPS_BY_LEVEL"] = path
r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", path],
                   env=env, capture_output=True, text=True)
okB = badB = None
try:
    okB, badB = map(int, r.stdout.strip().splitlines()[-1].split())
    print(f"B con gruppi (impostati all'avvio):   split conformi {okB}, NON conformi {badB}")
except Exception:
    print("B: processo figlio fallito")
    print(r.stdout)
    print(r.stderr)
# conferma dalla DLL: il WARNING "[RF_GROUPS] attivi: ..." (XGBoost lo inoltra
# a Python, su stdout o stderr a seconda della versione)
rf_lines = [ln for ln in (r.stdout + "\n" + r.stderr).splitlines() if "[RF_GROUPS]" in ln]
print("B messaggio DLL:", rf_lines[-1].strip() if rf_lines else "(nessuno)")

# D) convivenza con l'XGBoost standard nello stesso processo
import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
os.environ["XGB_RF_GROUPS_BY_LEVEL"] = path
okD, badD = train_and_count(xgb)
okD2, badD2 = train_and_count(rf)   # e xgboost_rf continua a rispettarla
libs_differ = lib_file(xgb) != lib_file(rf)
print(f"D xgboost standard, variabile attiva: split conformi {okD}, NON conformi {badD}")
print(f"D xgboost_rf dopo l'import standard:  split conformi {okD2}, NON conformi {badD2}")
print(f"D librerie native: {lib_file(rf)} | {lib_file(xgb)}")

# E) passaggio del modello xgboost_rf -> classe standard (percorso del trainer)
X, y = make_data()
m_rf = rf.XGBRegressor(n_estimators=60, **PARAMS).fit(X, y)
p_rf = m_rf.predict(X)
m_std = xgb.XGBRegressor()
m_std.load_model(bytearray(m_rf.get_booster().save_raw()))
p_std = m_std.predict(X)
# stesso modello, due librerie compilate in modo diverso: atteso identico al
# bit; si tollera solo un arrotondamento float32 (1e-6) e si stampa lo scarto
max_diff = float(np.max(np.abs(p_rf.astype(np.float64) - p_std.astype(np.float64))))
same = max_diff <= 1e-6
okE, badE = violations(m_std.get_booster())
print(f"E modello trasferito nella classe standard: scarto massimo predizioni {max_diff:.3g}, "
      f"split NON conformi {badE}")
os.environ.pop("XGB_RF_GROUPS_BY_LEVEL")

print(f"{RF_MOD} {rf.__version__}  |  xgboost {xgb.__version__}")

passA = (badA == 0 and okA > 0)
passB = (badB == 0 and (okB or 0) > 0)
errors = []
if not passA:
    if passB:
        errors.append("A: i gruppi funzionano solo se la variabile c'e' gia' all'avvio")
    else:
        errors.append("A/B: la regola per livello non e' rispettata")
elif not passB:
    errors.append("B: prova A ok ma B no (risultato incoerente)")
if badC == 0:
    errors.append("C: anche senza gruppi nessuna violazione (la prova non discrimina)")
if leak:
    errors.append("L: il pacchetto rinominato importa ancora 'xgboost'")
if badD == 0:
    errors.append("D: anche lo standard rispetta i gruppi (le due librerie non sono separate)")
if badD2 != 0:
    errors.append("D: xgboost_rf ha perso i gruppi dopo l'import dello standard")
if not libs_differ:
    errors.append("D: le due versioni usano la stessa libreria nativa")
if not same or badE != 0:
    errors.append("E: il modello trasferito nella classe standard non e' identico")
if errors:
    for e in errors:
        print("ERRORE", e)
    sys.exit("SELFTEST FALLITO")
print("SELFTEST OK: xgboost_rf rispetta i gruppi (anche impostati dopo l'import), "
      "convive con xgboost standard e i modelli si trasferiscono identici")
