#!/usr/bin/env python3
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR  = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("DATABASE_URL",      "postgresql+asyncpg://localhost/stub")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql+psycopg://localhost/stub")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

TILES_TO_CHECK = ["krasnodar_01", "kursk_01"]
NPZ_DIR  = PROJECT_ROOT / "backend/debug/runs/real_tiles"


def pct(mask) -> str:
    a = np.asarray(mask, dtype=bool)
    return f"{a.sum():>6d} px  ({100.0 * a.mean():.3f}%)"


def main():
    try:
        from core.config import getsettings
        settings = getsettings()
    except Exception:
        from core.config import get_settings
        settings = get_settings()

    import inspect
    from rasterio.transform import from_bounds

    _pp = __import__("processing.fields.postprocess", fromlist=["*"])
    rp = getattr(_pp, "run_postprocess", None) or getattr(_pp, "runpostprocess", None)
    if rp is None:
        print("❌ run_postprocess not found"); return

    pheno_mod = __import__("processing.fields.phenoclassify", fromlist=["*"])
    WATER   = getattr(pheno_mod, "WATER")
    CROP    = getattr(pheno_mod, "CROP")
    FOREST  = getattr(pheno_mod, "FOREST")
    BUILTUP = getattr(pheno_mod, "BUILTUP", None)

    sig_params = set(inspect.signature(rp).parameters.keys())

    for tile_id in TILES_TO_CHECK:
        npz_path = NPZ_DIR / f"{tile_id}.npz"
        if not npz_path.exists():
            print(f"❌ {tile_id}.npz not found"); continue

        z = np.load(npz_path)
        print(f"\n{'═'*60}")
        print(f"  TILE: {tile_id}")
        print(f"{'═'*60}")
        print(f"📦 NPZ keys: {sorted(z.keys())}")

        bbox      = tuple(map(float, z["bbox"].tolist()))
        maxndvi   = z["maxndvi"].astype(np.float32)
        ndvistd   = z["ndvistd"].astype(np.float32)
        edgecomp  = z["edgecomposite"].astype(np.float32)
        H, W      = maxndvi.shape
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], W, H)

        candidatemask = z["candidatemask"].astype(bool) if "candidatemask" in z \
            else np.zeros((H, W), dtype=bool)
        watermask = z["watermask"].astype(bool) if "watermask" in z \
            else np.zeros((H, W), dtype=bool)
        ndwi = z["ndwi"].astype(np.float32) if "ndwi" in z \
            else np.zeros((H, W), dtype=np.float32)

        classes = np.zeros((H, W), dtype=np.uint8)
        classes[watermask]     = WATER
        classes[candidatemask] = CROP

        print(f"\n🔍 Input:")
        print(f"   candidatemask : {pct(candidatemask)}")
        print(f"   watermask     : {pct(watermask)}")
        print(f"   ndvistd nonzero: {np.count_nonzero(ndvistd)}")
        print(f"   edgecomp nonzero: {np.count_nonzero(edgecomp)}")

        kw = {}
        def _a(key, val):
            if key in sig_params:
                kw[key] = val

        _a("candidatemask",      candidatemask.copy())
        _a("candidate_mask",     candidatemask.copy())
        _a("watermask",          watermask)
        _a("water_mask",         watermask)
        _a("classes",            classes)
        _a("ndvi",               maxndvi)
        _a("ndwi",               ndwi)
        _a("cfg",                settings)
        _a("edgecomposite",      edgecomp)
        _a("edge_composite",     edgecomp)
        _a("ndvistd",            ndvistd)
        _a("ndvi_std",           ndvistd)
        _a("worldcovermask",     None)
        _a("worldcover_mask",    None)
        _a("bbox",               bbox)
        _a("tiletransform",      transform)
        _a("tile_transform",     transform)
        _a("outshape",           (H, W))
        _a("out_shape",          (H, W))
        _a("crsepsg",            4326)
        _a("crs_epsg",           4326)
        _a("returndebugsteps",   True)
        _a("return_debug_steps", True)

        try:
            out = rp(**kw)
        except Exception as e:
            import traceback
            print(f"\n❌ run_postprocess raised: {e}")
            traceback.print_exc()
            continue

        final_mask  = out[0] if isinstance(out, tuple) else out
        debug_masks = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
        debug_stats = out[2] if isinstance(out, tuple) and len(out) > 2 else {}

        STEPS = [
            "step00candidateinitial",
            "step01roadmask",
            "step02forestmask",
            "step02bbuiltupmask",
            "step02cworldcoverweakprior",
            "step03barriermask",
            "step03bfieldcandidate",
            "step03ccropsoftmask",
            "fieldmaskboundary",
            "step04afterbarrier",
            "step05afterclean",
            "step06aftergrow",
            "step07aftergapclose",
            "step07bsmallcomponents",
            "step08afterinfill",
            "step09aftermerge",
            "step10afterwatershed",
            "step11aftersmallremove",
            "step12afterworldcoverreapply",
        ]

        print(f"\n📊 Pipeline steps:")
        if isinstance(debug_masks, dict):
            for step in STEPS:
                if step in debug_masks:
                    arr = np.asarray(debug_masks[step], dtype=bool)
                    bar = "█" * max(1, int(arr.mean() * 40)) if arr.any() else ""
                    print(f"   {step:<38s} {pct(arr)}  {bar}")
            # шаги которых нет в STEPS но есть в debug_masks
            extra = [k for k in debug_masks if k not in STEPS]
            if extra:
                print(f"   (extra keys: {extra})")
        else:
            print(f"   debug_masks type={type(debug_masks)}, value={debug_masks}")

        print(f"\n🏁 final_mask : {pct(np.asarray(final_mask, dtype=bool))}")

        if isinstance(debug_stats, dict) and debug_stats:
            print("\n📈 debug_stats:")
            for k, v in debug_stats.items():
                if isinstance(v, dict):
                    print(f"   {k:<38s} px={v.get('pixels','?'):>6}  "
                          f"cov={v.get('coverageratio', 0)*100:.2f}%")


if __name__ == "__main__":
    main()
