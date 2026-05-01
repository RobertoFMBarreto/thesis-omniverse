# dataset_generator_v1.py
# Versão sem rotação — baseline limpa
# Corre FORA do container

import numpy as np
from pathlib import Path
import json

COMPATIBLE = {
    "rectangle": 0,
    "square":    1,
    "circle":    2,
    "star":      3,
}

PIECES    = list(COMPATIBLE.keys())
N_CAVITIES = 4

OUT_DIR = Path("dataset_v1")
OUT_DIR.mkdir(exist_ok=True)

# Carrega point clouds
pieces = {}
for name in PIECES:
    path = Path(f"pc_{name}.npy")
    if not path.exists():
        print(f"[AVISO] {path} nao encontrado")
        continue
    pieces[name] = np.load(str(path))
    print(f"[OK] pc_{name}: {pieces[name].shape}")

cavities = {}
for idx in range(N_CAVITIES):
    path = Path(f"pc_cavity_{idx}.npy")
    if not path.exists():
        print(f"[AVISO] {path} nao encontrado")
        continue
    cavities[idx] = np.load(str(path))
    print(f"[OK] pc_cavity_{idx}: {cavities[idx].shape}")

# Gera todos os pares
pairs   = []
pair_id = 0

for piece_name, pc_piece in pieces.items():
    compatible_cav = COMPATIBLE[piece_name]

    for cav_idx, pc_cav in cavities.items():
        label = 1 if cav_idx == compatible_cav else 0

        pair_data = {
            "piece_pc":   pc_piece.astype(np.float32),
            "cavity_pc":  pc_cav.astype(np.float32),
            "label":      np.int32(label),
            "piece_name": piece_name,
            "cavity_idx": cav_idx,
            "rotation":   0,
        }
        np.save(str(OUT_DIR / f"pair_{pair_id:05d}.npy"),
                pair_data, allow_pickle=True)

        pairs.append({
            "id":        pair_id,
            "piece":     piece_name,
            "cavity":    cav_idx,
            "label":     label,
        })
        pair_id += 1

# Guarda índice
with open(str(OUT_DIR / "index.json"), "w") as f:
    json.dump(pairs, f, indent=2)

# Sumário
total     = len(pairs)
positivos = sum(1 for p in pairs if p["label"] == 1)
negativos = total - positivos

print(f"\n=== Dataset v1 (sem rotacao) ===")
print(f"  Total pares:  {total}")
print(f"  Positivos:    {positivos}  ({100*positivos/total:.1f}%)")
print(f"  Negativos:    {negativos}  ({100*negativos/total:.1f}%)")
print(f"\n  Pares por peça:")
for name in PIECES:
    pos = sum(1 for p in pairs if p["piece"]==name and p["label"]==1)
    neg = sum(1 for p in pairs if p["piece"]==name and p["label"]==0)
    print(f"    {name}: {pos} positivo + {neg} negativos")