"""
rf_patch.py - applica a XGBoost la modifica "gruppi di feature per livello".

Uso:  python rf_patch.py <cartella sorgenti xgboost>

Cosa cambia (solo src/common/random.h, classe ColumnSampler):
- se la variabile d'ambiente XGB_RF_GROUPS_BY_LEVEL contiene il percorso di
  un file di testo, ogni riga del file e' un gruppo di indici di feature
  (separati da virgola). Al livello di profondita' d dell'albero sono
  ammesse SOLO le feature del gruppo (d mod numero_gruppi):
      riga 1 -> radice (livello 0), riga 2 -> livello 1, riga 3 -> livello 2,
      poi si ricomincia (livello 3 -> riga 1, ...).
- senza la variabile d'ambiente il comportamento e' identico all'originale.
- il filtro si applica DOPO il campionamento colsample_bytree/bylevel/bynode;
  se il campionamento per albero ha escluso tutto il gruppo di un livello, a
  quel livello si usa il gruppo intero.

Lo script cerca punti di aggancio testuali: se la versione di XGBoost non
li contiene si ferma con errore, senza modificare nulla a meta'.
"""
import sys, os, re

src = os.path.join(sys.argv[1], "src", "common", "random.h")
s = open(src, encoding="utf-8").read()
orig = s

def must_replace(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        sys.exit(f"[rf_patch] ERRORE: punto di aggancio trovato {n} volte (attese {count}):\n{old}")
    s = s.replace(old, new)

# 1) include
must_replace('#include <vector>\n',
             '#include <vector>\n'
             '#include <cstdlib>   // [RF_GROUPS] getenv\n'
             '#include <fstream>   // [RF_GROUPS]\n'
             '#include <sstream>   // [RF_GROUPS]\n'
             '#include <string>    // [RF_GROUPS]\n')

# 2) membri e funzioni di supporto, subito dopo ctx_
must_replace('  Context const* ctx_;\n',
'''  Context const* ctx_;

  // ---- [RF_GROUPS] gruppi di feature per livello di profondita' -------------
  // rf_mask_[g][f] = 1 se la feature f appartiene al gruppo g. Vuoto = modifica
  // inattiva (comportamento originale).
  std::vector<std::vector<char>> rf_mask_;
  std::map<int, std::shared_ptr<HostDeviceVector<bst_feature_t>>> rf_level_cache_;

  void RfLoadGroups(int64_t num_col) {
    rf_mask_.clear();
    rf_level_cache_.clear();
    const char* path = std::getenv("XGB_RF_GROUPS_BY_LEVEL");
    if (path == nullptr || path[0] == '\\0') {
      return;
    }
    std::ifstream fin(path);
    if (!fin) {
      LOG(FATAL) << "[RF_GROUPS] impossibile aprire XGB_RF_GROUPS_BY_LEVEL=" << path;
    }
    std::string line;
    while (std::getline(fin, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      for (auto& ch : line) {
        if (ch == ',' || ch == ';' || ch == '\\t' || ch == '\\r') {
          ch = ' ';
        }
      }
      std::istringstream ss(line);
      std::vector<char> mask(static_cast<std::size_t>(num_col), 0);
      long long v = 0;
      bool any = false;
      while (ss >> v) {
        if (v >= 0 && v < num_col) {
          mask[static_cast<std::size_t>(v)] = 1;
          any = true;
        }
      }
      if (any) {
        rf_mask_.push_back(std::move(mask));
      }
    }
    if (rf_mask_.empty()) {
      LOG(FATAL) << "[RF_GROUPS] nessun gruppo valido in " << path;
    }
  }

  std::shared_ptr<HostDeviceVector<bst_feature_t>> RfFilter(
      std::shared_ptr<HostDeviceVector<bst_feature_t>> base, int depth) {
    auto const& mask = rf_mask_[static_cast<std::size_t>(depth) % rf_mask_.size()];
    auto const& h_base = base->ConstHostVector();
    auto out = std::make_shared<HostDeviceVector<bst_feature_t>>();
    if (!ctx_->Device().IsSycl()) {
      out->SetDevice(ctx_->Device());
    }
    auto& h_out = out->HostVector();
    for (auto f : h_base) {
      if (static_cast<std::size_t>(f) < mask.size() && mask[f]) {
        h_out.push_back(f);
      }
    }
    if (h_out.empty()) {
      // il campionamento per albero ha escluso tutto il gruppo: gruppo intero
      for (std::size_t f = 0; f < mask.size(); ++f) {
        if (mask[f]) {
          h_out.push_back(static_cast<bst_feature_t>(f));
        }
      }
    }
    return out;
  }
  // ---- [RF_GROUPS] fine ------------------------------------------------------
''')

# 3) Init: carica i gruppi
must_replace('    ctx_ = ctx;\n',
             '    ctx_ = ctx;\n'
             '    RfLoadGroups(num_col);  // [RF_GROUPS]\n')

# 4) Reset: svuota la cache per livello
must_replace('    feature_set_level_.clear();\n',
             '    feature_set_level_.clear();\n'
             '    rf_level_cache_.clear();  // [RF_GROUPS]\n')

# 5) GetFeatureSet: l'originale diventa GetFeatureSetBase, il nuovo filtra
must_replace('  std::shared_ptr<HostDeviceVector<bst_feature_t>> GetFeatureSet(int depth) {\n',
'''  // [RF_GROUPS] insieme di feature del nodo, ristretto al gruppo del livello
  std::shared_ptr<HostDeviceVector<bst_feature_t>> GetFeatureSet(int depth) {
    if (rf_mask_.empty()) {
      return GetFeatureSetBase(depth);
    }
    if (colsample_bynode_ == 1.0f) {
      auto it = rf_level_cache_.find(depth);
      if (it != rf_level_cache_.end()) {
        return it->second;
      }
      auto res = RfFilter(GetFeatureSetBase(depth), depth);
      rf_level_cache_[depth] = res;
      return res;
    }
    return RfFilter(GetFeatureSetBase(depth), depth);
  }

  std::shared_ptr<HostDeviceVector<bst_feature_t>> GetFeatureSetBase(int depth) {
''')

open(src, "w", encoding="utf-8").write(s)
print(f"[rf_patch] OK: {src} modificato ({len(s) - len(orig)} caratteri aggiunti)")
