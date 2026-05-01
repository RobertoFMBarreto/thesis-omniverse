import asyncio
import math

# ── utilitario de movimento da camara ────────────────────────────────────────

def set_camera_pose(x, y, z, rot_z_deg=0.0):
    import omni.usd
    from pxr import UsdGeom, Gf

    stage     = omni.usd.get_context().get_stage()
    cam_prim  = stage.GetPrimAtPath("/World/Camera")
    xformable = UsdGeom.Xformable(cam_prim)
    ops_dict  = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}

    ops_dict["xformOp:translate"].Set(Gf.Vec3d(x, y, z))

    half = math.radians(rot_z_deg) / 2
    quat = Gf.Quatd(math.cos(half), 0.0, 0.0, math.sin(half))

    if "xformOp:orient" in ops_dict:
        ops_dict["xformOp:orient"].Set(quat)
    elif "xformOp:rotateXYZ" in ops_dict:
        ops_dict["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 0.0, rot_z_deg))
    elif "xformOp:rotateZ" in ops_dict:
        ops_dict["xformOp:rotateZ"].Set(rot_z_deg)

    print(f"[cam] pos=({x}, {y}, {z})  rotZ={rot_z_deg}°")


# ── fase 1 — peca ─────────────────────────────────────────────────────────────

async def phase1_capture_piece():
    import omni.replicator.core as rep
    import numpy as np
    import cv2
    from pathlib import Path

    OUT = Path("/tmp/shape_insertion/data/raw")
    OUT.mkdir(parents=True, exist_ok=True)

    CAM_Z     = 1.00
    TABLE_TOP = 0.40
    OBJ_THRESH = (CAM_Z - TABLE_TOP) - 0.008

    set_camera_pose(-0.25, 0.45, CAM_Z, rot_z_deg=0.0)

    rp       = rep.create.render_product("/World/Camera", (640, 480))
    depth_an = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_an   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_an.attach([rp])
    rgb_an.attach([rp])
    await rep.orchestrator.step_async(rt_subframes=8)

    depth = depth_an.get_data()
    rgb   = rgb_an.get_data()
    d     = np.nan_to_num(depth.astype(np.float32), nan=0, posinf=0, neginf=0)

    above = (d > 0.05) & (d < OBJ_THRESH)
    print(f"[fase 1] obj_thresh={OBJ_THRESH:.3f}m  pixels acima mesa: {above.sum()}")

    if above.sum() == 0:
        print("[AVISO] peca nao detetada")
        cv2.imwrite(str(OUT/"phase1_rgb.png"),
                    cv2.cvtColor(rgb[:,:,:3], cv2.COLOR_RGB2BGR))
        return None

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        above.astype(np.uint8) * 255)

    blobs = sorted(
        [(int(centroids[i][0]), int(centroids[i][1]), stats[i, cv2.CC_STAT_AREA])
         for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 500],
        key=lambda b: b[2], reverse=True
    )
    print(f"[fase 1] blobs validos: {len(blobs)}")
    for i, (cx, cy, area) in enumerate(blobs):
        print(f"  blob {i}: area={area}px  centro=({cx},{cy})")

    if not blobs:
        print("[AVISO] nenhum blob valido")
        return None

    cx, cy, area = blobs[0]
    r = 32
    h, w = d.shape
    y1,y2 = max(0,cy-r), min(h,cy+r)
    x1,x2 = max(0,cx-r), min(w,cx+r)
    crop = d[y1:y2, x1:x2].copy()
    crop = np.pad(crop, ((0,64-crop.shape[0]),(0,64-crop.shape[1])), mode='edge')
    crop = crop[:64,:64]
    mn,mx = crop.min(), crop.max()
    piece_n = (crop-mn)/(mx-mn+1e-8)

    def to_vis(arr):
        return cv2.applyColorMap((arr*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    debug = rgb[:,:,:3].copy()
    debug[above] = (debug[above]*0.3 + np.array([255,60,60])*0.7).astype(np.uint8)
    cv2.circle(debug, (cx,cy), 6, (255,255,0), -1)
    cv2.rectangle(debug, (x1,y1), (x2,y2), (255,255,0), 2)

    cv2.imwrite(str(OUT/"phase1_rgb.png"),   cv2.cvtColor(rgb[:,:,:3], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT/"phase1_debug.png"), cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT/"crop_piece.png"),   to_vis(piece_n))
    np.save(str(OUT/"piece_crop.npy"), piece_n.astype(np.float32))

    print(f"[OK] peca capturada  centro=({cx},{cy})  area={area}px")
    print(f"\ndocker cp <container>:{OUT}/phase1_debug.png ./")
    print(f"docker cp <container>:{OUT}/crop_piece.png ./")

    return piece_n


# ── fase 2 — cavidades ────────────────────────────────────────────────────────

async def phase2_capture_cavities():
    import omni.replicator.core as rep
    import numpy as np
    import cv2
    from pathlib import Path

    OUT = Path("/tmp/shape_insertion/data/raw")
    OUT.mkdir(parents=True, exist_ok=True)

    CAM_Z      = 1.00
    TABLE_TOP  = 0.40
    OBJ_THRESH = (CAM_Z - TABLE_TOP) - 0.008

    set_camera_pose(0.30, 0.00, CAM_Z, rot_z_deg=-90.0)

    rp       = rep.create.render_product("/World/Camera", (640, 480))
    depth_an = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_an   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_an.attach([rp])
    rgb_an.attach([rp])
    await rep.orchestrator.step_async(rt_subframes=8)

    depth = depth_an.get_data()
    rgb   = rgb_an.get_data()
    d     = np.nan_to_num(depth.astype(np.float32), nan=0, posinf=0, neginf=0)

    above = (d > 0.05) & (d < OBJ_THRESH)
    print(f"[fase 2] obj_thresh={OBJ_THRESH:.3f}m  pixels acima mesa: {above.sum()}")

    if above.sum() == 0:
        print("[AVISO] nada detetado")
        cv2.imwrite(str(OUT/"phase2_rgb.png"),
                    cv2.cvtColor(rgb[:,:,:3], cv2.COLOR_RGB2BGR))
        return []

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        above.astype(np.uint8) * 255)

    blobs = sorted(
        [(int(centroids[i][0]), int(centroids[i][1]), stats[i, cv2.CC_STAT_AREA])
         for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 300],
        key=lambda b: b[2], reverse=True
    )
    print(f"[fase 2] blobs validos: {len(blobs)}")
    for i, (cx, cy, area) in enumerate(blobs):
        print(f"  blob {i}: area={area}px  centro=({cx},{cy})")

    def to_vis(arr):
        return cv2.applyColorMap((arr*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    cavity_crops = []
    debug = rgb[:,:,:3].copy()
    debug[above] = (debug[above]*0.3 + np.array([60,60,255])*0.7).astype(np.uint8)

    for idx, (cx, cy, area) in enumerate(blobs):
        r = 32
        h, w = d.shape
        y1,y2 = max(0,cy-r), min(h,cy+r)
        x1,x2 = max(0,cx-r), min(w,cx+r)
        crop = d[y1:y2, x1:x2].copy()
        crop = np.pad(crop,((0,64-crop.shape[0]),(0,64-crop.shape[1])),mode='edge')
        crop = crop[:64,:64]
        mn,mx = crop.min(), crop.max()
        crop_n = (crop-mn)/(mx-mn+1e-8)
        cavity_crops.append(crop_n)

        cv2.circle(debug, (cx,cy), 5, (255,255,0), -1)
        cv2.rectangle(debug, (x1,y1), (x2,y2), (255,255,0), 2)
        cv2.putText(debug, str(idx), (x1+2,y1+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        cv2.imwrite(str(OUT/f"crop_cavity_{idx}.png"), to_vis(crop_n))
        np.save(str(OUT/f"cavity_crop_{idx}.npy"), crop_n.astype(np.float32))
        print(f"  [OK] cavidade {idx}: centro=({cx},{cy})  area={area}px")

    cv2.imwrite(str(OUT/"phase2_rgb.png"),
                cv2.cvtColor(rgb[:,:,:3], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT/"phase2_debug.png"),
                cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))

    if cavity_crops:
        cv2.imwrite(str(OUT/"cavities_grid.png"),
                    np.hstack([to_vis(c) for c in cavity_crops]))

    print(f"\n[OK] {len(cavity_crops)} cavidades capturadas")
    print(f"\ndocker cp <container>:{OUT}/phase2_debug.png ./")
    print(f"docker cp <container>:{OUT}/cavities_grid.png ./")

    return cavity_crops


# ── pipeline completo ─────────────────────────────────────────────────────────

async def run_pipeline():
    print("=== FASE 1: captura da peca ===")
    piece = await phase1_capture_piece()

    print("\n=== FASE 2: captura das cavidades ===")
    cavities = await phase2_capture_cavities()

    print(f"\n=== RESULTADO ===")
    print(f"  peca:      {'OK' if piece is not None else 'FALHOU'}")
    print(f"  cavidades: {len(cavities)} capturadas")

asyncio.ensure_future(run_pipeline())