"""
rf_selftest.py - verifica che l'XGBoost compilato rispetti i gruppi per livello.

Addestra un piccolo modello su dati casuali con 3 gruppi (feature 0-2, 3-5,
6-8) e controlla, nodo per nodo, che a profondita' d sia usata solo una
feature del gruppo d mod 3. Poi ripete SENZA la variabile d'ambiente e
controlla che la regola NON sia piu' rispettata (prova che il filtro dipende
davvero dalla modifica). Esce con errore se qualcosa non torna.

[RF_ENV] Tre prove, per distinguere le cause in caso di errore:
  A) variabile impostata DOPO "import xgboost", nello stesso processo: e' il
     caso d'uso reale del trainer (gruppi diversi da un train all'altro).
  B) variabile impostata PRIMA dell'avvio, in un processo figlio: se B passa
     e A no, la DLL legge solo l'ambiente presente al caricamento (runtime C
     statico); se falliscono entrambe, il filtro non agisce sugli split.
  C) senza variabile: la regola NON deve valere (la prova discrimina).
Uso interno: "python rf_selftest.py --child <file gruppi>" esegue solo B.
"""
import os, sys, subprocess, tempfile

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


def train_and_count():
    import xgboost as xgb
    X, y = make_data()
    dm = xgb.DMatrix(X, label=y)
    return violations(xgb.train(PARAMS, dm, 60))


# ---- processo figlio: prova B (variabile gia' presente all'avvio) ----------
if len(sys.argv) >= 3 and sys.argv[1] == "--child":
    ok, bad = train_and_count()
    print(f"{ok} {bad}")
    sys.exit(0)

path = os.path.join(tempfile.gettempdir(), "rf_groups_selftest.txt")
with open(path, "w") as fh:
    fh.write("\n".join(",".join(map(str, g)) for g in GROUPS))

import xgboost as xgb  # noqa: E402  (import PRIMA di impostare la variabile: prova A)

# A) variabile impostata dopo l'import
os.environ["XGB_RF_GROUPS_BY_LEVEL"] = path
okA, badA = train_and_count()
print(f"A con gruppi (impostati dopo import): split conformi {okA}, NON conformi {badA}")

# C) senza variabile
os.environ.pop("XGB_RF_GROUPS_BY_LEVEL")
okC, badC = train_and_count()
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

print("xgboost", xgb.__version__)

passA = (badA == 0 and okA > 0)
passB = (badB == 0 and (okB or 0) > 0)
if not passA:
    if passB:
        sys.exit("SELFTEST FALLITO: i gruppi funzionano solo se la variabile c'e' gia' "
                 "all'avvio del processo (la DLL non vede os.environ dopo l'import)")
    sys.exit("SELFTEST FALLITO: la regola per livello non e' rispettata nemmeno con la "
             "variabile presente all'avvio (il filtro non agisce sugli split)")
if not passB:
    sys.exit("SELFTEST FALLITO: prova A ok ma prova B no (risultato incoerente)")
if badC == 0:
    sys.exit("SELFTEST DUBBIO: anche senza gruppi nessuna violazione (la prova non discrimina)")
print("SELFTEST OK: gruppi per livello attivi solo con XGB_RF_GROUPS_BY_LEVEL, "
      "anche se impostata dopo l'import")
