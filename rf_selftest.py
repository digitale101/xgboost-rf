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

[RF_CUDA] Build CUDA (workflow build-cuda.yml). Due opzioni, combinabili:
  --expect-cuda  la libreria DEVE essere compilata con CUDA
                 (build_info()["USE_CUDA"]). Usata dal workflow CUDA: il
                 runner di GitHub non ha GPU, quindi li' le prove A-E girano
                 sulla CPU (stesso sorgente patchato) e si controlla solo
                 che la build sia davvero CUDA.
  --gpu          prove SULLA GPU, da lanciare sul PC con la scheda:
     G1) con gruppi, device cuda: nessuno split fuori gruppo e booster
         addestrato davvero sulla GPU (fail_on_invalid_gpu_id + device
         letto da save_config: nessun ripiego silenzioso sulla CPU);
     G2) senza gruppi sulla GPU: la regola NON deve valere;
     G3) modello addestrato sulla GPU trasferito nella classe standard:
         predizioni identiche (scarto <= 1e-6).
  Esempio sul PC:  python rf_selftest.py --gpu --expect-cuda
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
# [RF_CUDA] opzioni (il processo figlio --child non le riceve)
EXPECT_CUDA = "--expect-cuda" in sys.argv
RUN_GPU = "--gpu" in sys.argv


def booster_device(bst):
    """[RF_CUDA] device di un booster APPENA addestrato (save_config)."""
    import json, re
    txt = bst.save_config()
    try:
        dev = json.loads(txt).get("learner", {}).get("generic_param", {}).get("device")
        if dev:
            return str(dev)
    except Exception:
        pass
    m = re.search(r'"device"\s*:\s*"([^"]+)"', txt)
    return m.group(1) if m else ""


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

# [RF_CUDA] build CUDA?
try:
    _bi = rf.build_info()
except Exception as _e:
    _bi = {"errore": str(_e)}
use_cuda = str(_bi.get("USE_CUDA")).strip().lower() in ("1", "true", "on", "yes")
print(f"CUDA {RF_MOD}: USE_CUDA={_bi.get('USE_CUDA')} CUDA_VERSION={_bi.get('CUDA_VERSION', '-')}")

# [RF_CUDA] prove sulla GPU (solo con --gpu, sul PC con la scheda)
gpu_errors = []
if RUN_GPU:
    PG = dict(PARAMS, device="cuda", fail_on_invalid_gpu_id=True)
    X, y = make_data()
    try:
        os.environ["XGB_RF_GROUPS_BY_LEVEL"] = path
        try:
            b_g1 = rf.train(PG, rf.DMatrix(X, label=y), 60)
        finally:
            os.environ.pop("XGB_RF_GROUPS_BY_LEVEL", None)
        dev1 = booster_device(b_g1)
        okG1, badG1 = violations(b_g1)
        print(f"G1 GPU con gruppi: device {dev1}, split conformi {okG1}, NON conformi {badG1}")
        if not dev1.lower().startswith("cuda"):
            gpu_errors.append(f"G1: addestrato su '{dev1 or 'non rilevato'}', non sulla GPU")
        if badG1 != 0 or okG1 == 0:
            gpu_errors.append("G1: sulla GPU la regola per livello non e' rispettata")

        b_g2 = rf.train(PG, rf.DMatrix(X, label=y), 60)
        dev2 = booster_device(b_g2)
        okG2, badG2 = violations(b_g2)
        print(f"G2 GPU senza gruppi: device {dev2}, split conformi {okG2}, NON conformi {badG2}")
        if not dev2.lower().startswith("cuda"):
            gpu_errors.append(f"G2: addestrato su '{dev2 or 'non rilevato'}', non sulla GPU")
        if badG2 == 0:
            gpu_errors.append("G2: anche senza gruppi nessuna violazione (la prova non discrimina)")

        os.environ["XGB_RF_GROUPS_BY_LEVEL"] = path
        try:
            m_g = rf.XGBRegressor(n_estimators=60, **PG).fit(X, y)
        finally:
            os.environ.pop("XGB_RF_GROUPS_BY_LEVEL", None)
        dev3 = booster_device(m_g.get_booster())
        p_g = m_g.predict(X)
        m_gs = xgb.XGBRegressor()
        m_gs.load_model(bytearray(m_g.get_booster().save_raw()))
        p_gs = m_gs.predict(X)
        d3 = float(np.max(np.abs(p_g.astype(np.float64) - p_gs.astype(np.float64))))
        okG3, badG3 = violations(m_gs.get_booster())
        print(f"G3 GPU -> classe standard: device {dev3}, scarto massimo {d3:.3g}, "
              f"split NON conformi {badG3}")
        if not dev3.lower().startswith("cuda"):
            gpu_errors.append(f"G3: addestrato su '{dev3 or 'non rilevato'}', non sulla GPU")
        if d3 > 1e-6 or badG3 != 0:
            gpu_errors.append("G3: il modello GPU trasferito nella classe standard non e' identico")
    except Exception as _e:
        gpu_errors.append(f"GPU: prova non eseguibile ({type(_e).__name__}: {_e})")

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
# [RF_CUDA]
if (EXPECT_CUDA or RUN_GPU) and not use_cuda:
    errors.append(f"CUDA: {RF_MOD} non e' compilato con CUDA (USE_CUDA={_bi.get('USE_CUDA')})")
errors.extend(gpu_errors)
if errors:
    for e in errors:
        print("ERRORE", e)
    sys.exit("SELFTEST FALLITO")
print("SELFTEST OK: xgboost_rf rispetta i gruppi (anche impostati dopo l'import), "
      "convive con xgboost standard e i modelli si trasferiscono identici"
      + (" | build CUDA" if use_cuda else "")
      + (" | prove GPU G1-G3 superate" if RUN_GPU else ""))
