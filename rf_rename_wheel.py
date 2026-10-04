"""
rf_rename_wheel.py - trasforma il pacchetto XGBoost compilato con la modifica
[RF_GROUPS] in un pacchetto a se' stante, installabile ACCANTO all'XGBoost
standard nello stesso ambiente Python.

Uso:  python rf_rename_wheel.py <cartella con il .whl di xgboost> <cartella di uscita>

Cosa fa sul .whl gia' costruito (non tocca il packager di XGBoost):
- nome del pacchetto pip:  xgboost      -> xgboost-rf
- cartella del modulo:     xgboost/     -> xgboost_rf/   (import xgboost_rf)
- libreria nativa:         xgboost.dll  -> xgboost_rf.dll, e i riferimenti
  "xgboost.dll" nei sorgenti Python aggiornati di conseguenza: le due dll
  hanno nomi diversi e possono stare caricate insieme nello stesso processo
- import assoluti interni ("import xgboost", "from xgboost.x import ...")
  riscritti su xgboost_rf; quelli relativi (".core", "..sklearn") non
  servono modifiche
- cartella .dist-info, METADATA e RECORD (hash sha256 ricalcolati)

Si ferma con errore se non trova la dll o nessun riferimento a "xgboost.dll":
meglio una build rossa che un pacchetto che carica la libreria sbagliata.
Gli import assoluti che non sa riscrivere in modo sicuro li elenca nel log.
"""
import base64, csv, hashlib, io, os, re, sys, zipfile

OLD_PKG, NEW_PKG = "xgboost", "xgboost_rf"
OLD_DIST, NEW_DIST = "xgboost", "xgboost_rf"          # nome nel file .whl / dist-info
NEW_NAME_META = "xgboost-rf"                          # "Name:" in METADATA
OLD_DLL, NEW_DLL = "xgboost.dll", "xgboost_rf.dll"


def fail(msg):
    sys.exit(f"[rf_rename] ERRORE: {msg}")


def rewrite_imports(text, relpath, notes):
    """Riscrive gli import assoluti del pacchetto. Ritorna (testo, n_modifiche)."""
    n = 0

    def sub(pattern, repl, s):
        nonlocal n
        s2, k = re.subn(pattern, repl, s, flags=re.M)
        n += k
        return s2

    # from xgboost import x / from xgboost.core import x
    text = sub(r"^(\s*)from xgboost((?:\.[\w.]+)?)(\s+import\b)", r"\1from xgboost_rf\2\3", text)
    # import xgboost as xgb / import xgboost.core as c
    text = sub(r"^(\s*)import xgboost((?:\.[\w.]+)?)(\s+as\s+)", r"\1import xgboost_rf\2\3", text)
    # import xgboost   (lega il nome "xgboost": lo si mantiene con un alias)
    text = sub(r"^(\s*)import xgboost\s*$", r"\1import xgboost_rf as xgboost", text)
    # import xgboost.sub  (senza alias): lega il nome "xgboost" al pacchetto
    # con il sottomodulo caricato; stesso effetto con xgboost_rf + alias
    text = sub(r"^(\s*)import xgboost(\.[\w.]+)\s*$",
               r"\1import xgboost_rf\2; import xgboost_rf as xgboost", text)
    # moduli nominati come stringa (import_module("xgboost.x"), ecc.): non
    # riscritti, elencati nel log (i nomi di file .dll/.so non contano)
    for m in re.finditer(r"""["']xgboost(\.[\w.]+)["']""", text):
        if m.group(1).lower() in (".dll", ".so", ".dylib"):
            continue
        notes.append(f"{relpath}: stringa con nome modulo (non riscritta): {m.group(0)}")
    return text, n


def record_line(path, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return [path, f"sha256={digest}", str(len(data))]


def main():
    if len(sys.argv) != 3:
        fail("uso: python rf_rename_wheel.py <cartella_whl> <cartella_uscita>")
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    whls = [f for f in os.listdir(src_dir)
            if f.endswith(".whl") and f.startswith(OLD_DIST + "-")]
    if len(whls) != 1:
        fail(f"attesa una sola wheel {OLD_DIST}-*.whl in {src_dir}, trovate: {whls}")
    src = os.path.join(src_dir, whls[0])
    # xgboost-3.2.0-py3-none-win_amd64.whl -> version, tag
    m = re.match(rf"^{OLD_DIST}-([^-]+)-(.+)\.whl$", whls[0])
    if not m:
        fail(f"nome wheel non riconosciuto: {whls[0]}")
    version, tag = m.group(1), m.group(2)
    old_info = f"{OLD_DIST}-{version}.dist-info/"
    new_info = f"{NEW_DIST}-{version}.dist-info/"

    files = {}
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            if not info.is_dir():
                files[info.filename] = z.read(info.filename)

    out = {}
    notes = []
    n_imports = n_dllrefs = 0
    dll_found = False
    for name, data in files.items():
        if name.startswith(old_info):
            if name.endswith("/RECORD"):
                continue  # rigenerato sotto
            new_name = new_info + name[len(old_info):]
            if name.endswith("/METADATA"):
                txt = data.decode("utf-8")
                txt, k = re.subn(r"^Name: .*$", f"Name: {NEW_NAME_META}", txt, count=1, flags=re.M)
                if k != 1:
                    fail("riga 'Name:' non trovata in METADATA")
                data = txt.encode("utf-8")
            out[new_name] = data
            continue
        if not name.startswith(OLD_PKG + "/"):
            out[name] = data  # file fuori dal pacchetto (non previsti, copiati com'e')
            notes.append(f"file fuori dal pacchetto copiato invariato: {name}")
            continue
        rel = name[len(OLD_PKG) + 1:]
        new_name = NEW_PKG + "/" + rel
        if rel.lower().endswith("/" + OLD_DLL) or rel.lower() == OLD_DLL:
            new_name = new_name[: -len(OLD_DLL)] + NEW_DLL
            dll_found = True
        elif name.endswith(".py"):
            txt = data.decode("utf-8")
            txt, k = rewrite_imports(txt, name, notes)
            n_imports += k
            k2 = txt.count(OLD_DLL)
            if k2:
                txt = txt.replace(OLD_DLL, NEW_DLL)
                n_dllrefs += k2
            data = txt.encode("utf-8")
        out[new_name] = data

    if not dll_found:
        fail(f"{OLD_DLL} non trovata dentro la wheel")
    if n_dllrefs == 0:
        fail(f"nessun riferimento a '{OLD_DLL}' nei sorgenti Python: "
             "non saprei come far caricare la dll rinominata")

    # RECORD
    rec_name = new_info + "RECORD"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for name in sorted(out):
        w.writerow(record_line(name, out[name]))
    w.writerow([rec_name, "", ""])
    out[rec_name] = buf.getvalue().encode("utf-8")

    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"{NEW_DIST}-{version}-{tag}.whl")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        # dist-info per ultima, come da convenzione delle wheel
        order = [n for n in sorted(out) if not n.startswith(new_info)] + \
                [n for n in sorted(out) if n.startswith(new_info) and n != rec_name] + [rec_name]
        for name in order:
            z.writestr(name, out[name])

    print(f"[rf_rename] OK: {dst}")
    print(f"[rf_rename] import riscritti: {n_imports}, riferimenti dll aggiornati: {n_dllrefs}")
    for note in notes:
        print(f"[rf_rename] nota: {note}")


if __name__ == "__main__":
    main()
