import asyncio
import math

def set_camera_pose(x, y, z, rot_z_deg=0.0):
    import omni.usd
    from pxr import UsdGeom, Gf
    stage    = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/Camera")
    ops_dict = {op.GetOpName(): op
                for op in UsdGeom.Xformable(cam_prim).GetOrderedXformOps()}
    ops_dict["xformOp:translate"].Set(Gf.Vec3d(x, y, z))
    half = math.radians(rot_z_deg) / 2
    quat = Gf.Quatd(math.cos(half), 0.0, 0.0, math.sin(half))
    if "xformOp:orient" in ops_dict:
        ops_dict["xformOp:orient"].Set(quat)
    print(f"[cam] pos=({x}, {y}, {z})  rotZ={rot_z_deg}°")

def world_to_pixel(wx, wy, cam_x, cam_y, cam_z,
                   img_w=640, img_h=480,
                   focal_mm=24.0, aperture_mm=36.0):
    fov_h = 2 * math.atan((aperture_mm / 2) / focal_mm)
    fov_v = fov_h * (img_h / img_w)
    mpp_x = (2 * cam_z * math.tan(fov_h / 2)) / img_w
    mpp_y = (2 * cam_z * math.tan(fov_v / 2)) / img_h
    u = int(img_w/2 + (wx - cam_x) / mpp_x)
    v = int(img_h/2 - (wy - cam_y) / mpp_y)
    return u, v

BOARD_BOUNDS = {
    "min": (0.2385, -0.0405, 0.40),
    "max": (0.3385,  0.0445, 0.42),
    "top_z": 0.42,
}
PIECE_BOUNDS = {
    "rectangle": {
        "min": (-0.2709, 0.4388, 0.40),
        "max": (-0.2394, 0.4568, 0.43),
        "top_z": 0.43,
    }
}
CAM_Z     = 1.00
TABLE_TOP = 0.40

async def phase1_capture_piece():
    import omni.replicator.core as rep
    import numpy as np
    import cv2
    from pathlib import Path

    OUT = Path("/tmp/shape_insertion/data/raw")
    OUT.mkdir(parents=True, exist_ok=True)

    bounds   = PIECE_BOUNDS["rectangle"]
    cx_world = (bounds["min"][0] + bounds["max"][0]) / 2
    cy_world = (bounds["min"][1] + bounds["max"][1]) / 2

    set_camera_pose(cx_world, cy_world, CAM_Z, rot_z_deg=0.0)

    rp       = rep.create.render_product("/World/Camera", (640, 480))
    depth_an = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_an   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_an.attach([rp])
    rgb_an.attach([rp])
    await rep.orchestrator.step_async(rt_subframes=8)

    depth = depth_an.get_data()
    rgb   = rgb_an.get_data()
    d     = np.nan_to_num(depth.astype(np.float32), nan=0, posinf=0, neginf=0)

    OBJ_THRESH = (CAM_Z - TABLE_TOP) - 0.008
    piece_mask = (d > 0.05) & (d < OBJ_THRESH)

    px, py = world_to_pixel(cx_world, cy_world, cx_world, cy_world, CAM_Z)
    r = 32
    h, w = d.shape
    y1,y2 = max(0,py-r), min(h,py+r)
    x1,x2 = max(0,px-r), min(w,px+r)
    crop = d[y1:y2,x1:x2].copy()
    crop = np.pad(crop,((0,64-crop.shape[0]),(0,64-crop.shape[1])),mode='edge')
    crop = crop[:64,:64]
    mn,mx = crop.min(), crop.max()
    piece_n = (crop-mn)/(mx-mn+1e-8)

    def to_vis(a):
        return cv2.applyColorMap((a*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    debug = rgb[:,:,:3].copy()
    debug[piece_mask] = (debug[piece_mask]*0.3
                         + np.array([255,60,60])*0.7).astype(np.uint8)
    cv2.circle(debug,(px,py),6,(255,255,0),-1)
    cv2.rectangle(debug,(x1,y1),(x2,y2),(255,255,0),2)

    cv2.imwrite(str(OUT/"phase1_rgb.png"),   cv2.cvtColor(rgb[:,:,:3],cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT/"phase1_debug.png"), cv2.cvtColor(debug,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT/"crop_piece.png"),   to_vis(piece_n))
    np.save(str(OUT/"piece_crop.npy"), piece_n.astype(np.float32))

    print(f"[fase 1] peca OK  pixels={piece_mask.sum()}  centro=({px},{py})")
    return piece_n

async def phase2_capture_cavities():
    import omni.replicator.core as rep
    import numpy as np
    import cv2
    from pathlib import Path

    OUT = Path("/tmp/shape_insertion/data/raw")
    OUT.mkdir(parents=True, exist_ok=True)

    bounds   = BOARD_BOUNDS
    cx_world = (bounds["min"][0] + bounds["max"][0]) / 2
    cy_world = (bounds["min"][1] + bounds["max"][1]) / 2
    top_z    = bounds["top_z"]

    set_camera_pose(cx_world, cy_world, CAM_Z, rot_z_deg=-90.0)

    rp       = rep.create.render_product("/World/Camera", (640, 480))
    depth_an = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_an   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_an.attach([rp])
    rgb_an.attach([rp])
    await rep.orchestrator.step_async(rt_subframes=8)

    depth = depth_an.get_data()
    rgb   = rgb_an.get_data()
    d     = np.nan_to_num(depth.astype(np.float32), nan=0, posinf=0, neginf=0)

    # ROI da board
    corners = [
        (bounds["min"][0], bounds["min"][1]),
        (bounds["max"][0], bounds["min"][1]),
        (bounds["min"][0], bounds["max"][1]),
        (bounds["max"][0], bounds["max"][1]),
    ]
    px_corners = [world_to_pixel(wx,wy,cx_world,cy_world,CAM_Z)
                  for wx,wy in corners]
    us = [p[0] for p in px_corners]
    vs = [p[1] for p in px_corners]
    roi_x1,roi_x2 = max(0,min(us)-5), min(640,max(us)+5)
    roi_y1,roi_y2 = max(0,min(vs)-5), min(480,max(vs)+5)

    board_surface = CAM_Z - top_z   # 0.58m

    # ── parametros calibrados ─────────────────────────────────────────────────
    HOLE_MARGIN = 0.001   # 1mm — apanha estrela
    MIN_HOLE    = 50      # px — apanha estrela

    roi_d = d[roi_y1:roi_y2, roi_x1:roi_x2]
    hole_roi = (
            (roi_d > board_surface + HOLE_MARGIN) &
            (roi_d < board_surface + 0.05) &
            (roi_d > 0.05)
    )
    hole_mask = np.zeros(d.shape, dtype=bool)
    hole_mask[roi_y1:roi_y2, roi_x1:roi_x2] = hole_roi

    n_h,h_labels,h_stats,h_centroids = cv2.connectedComponentsWithStats(
        hole_mask.astype(np.uint8)*255)

    holes = sorted(
        [(int(h_centroids[i][0]), int(h_centroids[i][1]),
          h_stats[i,cv2.CC_STAT_AREA])
         for i in range(1,n_h)
         if h_stats[i,cv2.CC_STAT_AREA] >= MIN_HOLE],
        key=lambda h: h[2], reverse=True
    )

    print(f"[fase 2] board_surface={board_surface:.3f}m  "
          f"HOLE_MARGIN={HOLE_MARGIN*1000:.0f}mm  MIN_HOLE={MIN_HOLE}px")
    print(f"[fase 2] buracos encontrados: {len(holes)}")
    for i,(cx,cy,area) in enumerate(holes):
        print(f"  buraco {i}: area={area}px  centro=({cx},{cy})")

    def to_vis(a):
        return cv2.applyColorMap((a*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    cavity_crops = []
    debug = rgb[:,:,:3].copy()
    cv2.rectangle(debug,(roi_x1,roi_y1),(roi_x2,roi_y2),(0,255,0),2)
    debug[hole_mask] = (debug[hole_mask]*0.3
                        + np.array([60,60,255])*0.7).astype(np.uint8)

    for idx,(cx,cy,area) in enumerate(holes):
        r = 32
        h,w = d.shape
        y1,y2 = max(0,cy-r), min(h,cy+r)
        x1,x2 = max(0,cx-r), min(w,cx+r)
        crop = d[y1:y2,x1:x2].copy()
        crop = np.pad(crop,((0,64-crop.shape[0]),(0,64-crop.shape[1])),mode='edge')
        crop = crop[:64,:64]
        mn,mx = crop.min(), crop.max()
        crop_n = (crop-mn)/(mx-mn+1e-8)
        cavity_crops.append(crop_n)

        cv2.circle(debug,(cx,cy),5,(255,255,0),-1)
        cv2.rectangle(debug,(x1,y1),(x2,y2),(255,255,0),2)
        cv2.putText(debug,str(idx),(x1+2,y1+14),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)

        cv2.imwrite(str(OUT/f"crop_cavity_{idx}.png"), to_vis(crop_n))
        np.save(str(OUT/f"cavity_crop_{idx}.npy"), crop_n.astype(np.float32))
        print(f"  [OK] cavidade {idx}: centro=({cx},{cy})  area={area}px")

    cv2.imwrite(str(OUT/"phase2_rgb.png"),
                cv2.cvtColor(rgb[:,:,:3],cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT/"phase2_debug.png"),
                cv2.cvtColor(debug,cv2.COLOR_RGB2BGR))

    d_vis = np.zeros_like(d)
    m = (d>0.1)&(d<1.0)
    if m.any():
        d_vis[m] = (d[m]-d[m].min())/(d[m].max()-d[m].min()+1e-8)
    cv2.imwrite(str(OUT/"phase2_depth.png"), to_vis(d_vis))

    if cavity_crops:
        cv2.imwrite(str(OUT/"cavities_grid.png"),
                    np.hstack([to_vis(c) for c in cavity_crops]))

    print(f"\n[OK] {len(cavity_crops)} cavidades capturadas")
    print(f"\ndocker cp <container>:{OUT}/phase2_debug.png ./")
    print(f"docker cp <container>:{OUT}/cavities_grid.png ./")

    return cavity_crops

async def run_pipeline():
    print("=== FASE 1 ===")
    piece = await phase1_capture_piece()
    print("\n=== FASE 2 ===")
    cavities = await phase2_capture_cavities()
    print(f"\n=== RESULTADO: peca={'OK' if piece is not None else 'FALHOU'}"
          f"  cavidades={len(cavities)} ===")

asyncio.ensure_future(run_pipeline())