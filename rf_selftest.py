"""
rf_selftest.py - verifica che l'XGBoost compilato rispetti i gruppi per livello.

Addestra un piccolo modello su dati casuali con 3 gruppi (feature 0-2, 3-5,
6-8) e controlla, nodo per nodo, che a profondita' d sia usata solo una
feature del gruppo d mod 3. Poi ripete SENZA la variabile d'ambiente e
controlla che la regola NON sia piu' rispettata (prova che il filtro dipende
davvero dalla modifica). Esce con errore se qualcosa non torna.
"""
import os, sys, tempfile
import numpy as np
import xgboost as xgb

rng = np.random.default_rng(0)
X = rng.normal(size=(3000, 9))
y = X[:, 0] * X[:, 4] + X[:, 8] + 0.5 * X[:, 7] * X[:, 2] + rng.normal(size=3000) * 0.1
groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

def violations(bst):
    df = bst.trees_to_dataframe()
    bad = ok = 0
    for _, t in df.groupby("Tree"):
        t = t.set_index("ID")
        depth = {}
        root = t.index[t["Node"] == 0][0]
        stack = [(root, 0)]
        while stack:
            nid, d = stack.pop()
            row = t.loc[nid]
            if row["Feature"] == "Leaf":
                continue
            f = int(str(row["Feature"]).lstrip("f"))
            if f in groups[d % len(groups)]:
                ok += 1
            else:
                bad += 1
            stack.append((row["Yes"], d + 1))
            stack.append((row["No"], d + 1))
    return ok, bad

path = os.path.join(tempfile.gettempdir(), "rf_groups_selftest.txt")
with open(path, "w") as fh:
    fh.write("\n".join(",".join(map(str, g)) for g in groups))

params = {"max_depth": 3, "tree_method": "hist", "eta": 0.3, "seed": 1}
dm = xgb.DMatrix(X, label=y)

os.environ["XGB_RF_GROUPS_BY_LEVEL"] = path
ok, bad = violations(xgb.train(params, dm, 60))
print(f"con gruppi:   split conformi {ok}, NON conformi {bad}")

os.environ.pop("XGB_RF_GROUPS_BY_LEVEL")
ok2, bad2 = violations(xgb.train(params, dm, 60))
print(f"senza gruppi: split conformi {ok2}, NON conformi {bad2}")

print("xgboost", xgb.__version__)
if bad != 0 or ok == 0:
    sys.exit("SELFTEST FALLITO: la regola per livello non e' rispettata")
if bad2 == 0:
    sys.exit("SELFTEST DUBBIO: anche senza gruppi nessuna violazione (la prova non discrimina)")
print("SELFTEST OK: gruppi per livello attivi solo con XGB_RF_GROUPS_BY_LEVEL")
