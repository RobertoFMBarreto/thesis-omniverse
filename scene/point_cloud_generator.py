import asyncio
import math

async def generate_pointclouds_final():
    import omni.usd
    import omni.replicator.core as rep
    import numpy as np
    import cv2
    from pathlib import Path
    from pxr import UsdGeom

    OUT = Path("/tmp/shape_insertion/data/raw")
    OUT.mkdir(parents=True, exist_ok=True)

    N_POINTS     = 1024
    GLOBAL_SCALE = 0.05
    CAM_Z_PIECE  = 0.58
    CAM_Z_BOARD  = 1.00

    PIECES = {
        "rectangle": {"center": (-0.2552, 0.4478), "size": (0.0315, 0.0180)},
        "star":      {"center": (-0.3528, 0.4186), "size": (0.0171, 0.0163)},
        "square":    {"center": (-0.1468, 0.4119), "size": (0.0180, 0.0180)},
        "circle":    {"center": (-0.3121, 0.3876), "size": (0.0180, 0.0179)},
    }

    BOARD = {
        "center": (0.2885, 0.0020),
        "bounds": (0.2385, -0.0405, 0.3385, 0.0445),
    }

    # ── utilitarios ───────────────────────────────────────────────────────────

    def show_only(piece_name):
        stage = omni.usd.get_context().get_stage()
        for name in PIECES.keys():
            prim = stage.GetPrimAtPath(f"/World/{name}")
            if prim.IsValid():
                img = UsdGeom.Imageable(prim)
                if name == piece_name:
                    img.MakeVisible()
                else:
                    img.MakeInvisible()

    def show_all():
        stage = omni.usd.get_context().get_stage()
        for name in PIECES.keys():
            prim = stage.GetPrimAtPath(f"/World/{name}")
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeVisible()

    def set_camera(x, y, z, rot_z=0.0):
        stage    = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath("/World/Camera")
        ops_dict = {op.GetOpName(): op
                    for op in UsdGeom.Xformable(cam_prim).GetOrderedXformOps()}
        ops_dict["xformOp:translate"].Set(__import__('pxr').Gf.Vec3d(x, y, z))
        half = math.radians(rot_z) / 2
        import pxr.Gf as Gf
        quat = Gf.Quatd(math.cos(half), 0.0, 0.0, math.sin(half))
        if "xformOp:orient" in ops_dict:
            ops_dict["xformOp:orient"].Set(quat)

    def get_surface_depth(d, d_min, d_max):
        zone = d[(d > d_min) & (d < d_max)]
        if len(zone) == 0:
            return None
        hist, edges = np.histogram(zone, bins=np.arange(d_min, d_max, 0.001))
        return edges[np.argmax(hist)] + 0.0005

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

    def pixels_to_world(us, vs, cam_x, cam_y, cam_z,
                        img_w=640, img_h=480,
                        focal_mm=24.0, aperture_mm=36.0):
        fov_h = 2 * math.atan((aperture_mm / 2) / focal_mm)
        fov_v = fov_h * (img_h / img_w)
        mpp_x = (2 * cam_z * math.tan(fov_h / 2)) / img_w
        mpp_y = (2 * cam_z * math.tan(fov_v / 2)) / img_h
        wx = cam_x + (us - img_w / 2) * mpp_x
        wy = cam_y - (vs - img_h / 2) * mpp_y
        return wx, wy

    def build_pointcloud(d, mask, cam_x, cam_y, cam_z,
                         d_ref, global_scale, n_points=1024):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None

        wx_top, wy_top = pixels_to_world(xs.astype(float), ys.astype(float),
                                         cam_x, cam_y, cam_z)
        wz_top  = d_ref - d[ys, xs]
        top_pts = np.stack([wx_top, wy_top, wz_top], axis=1)
        cx_obj  = wx_top.mean()
        cy_obj  = wy_top.mean()
        z_max   = max(wz_top.max(), 0.001)

        wall_list = []
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if contours:
            main_c = max(contours, key=len)
            cpts   = main_c.reshape(-1, 2)
            cwx, cwy = pixels_to_world(cpts[:,0].astype(float),
                                       cpts[:,1].astype(float),
                                       cam_x, cam_y, cam_z)
            for level in range(9):
                wall_z   = z_max * level / 8
                wall_pts = np.stack(
                    [cwx, cwy, np.full(len(cwx), wall_z)], axis=1)
                wall_list.append(wall_pts)

        points = np.vstack([top_pts] + wall_list).astype(np.float32)
        points[:, 0] -= cx_obj
        points[:, 1] -= cy_obj
        points[:, 2] -= points[:, 2].min()
        points        /= global_scale

        replace = len(points) < n_points
        idx     = np.random.choice(len(points), n_points, replace=replace)
        return points[idx]

    rp       = rep.create.render_product("/World/Camera", (640, 480))
    depth_an = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_an   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_an.attach([rp])
    rgb_an.attach([rp])

    # ── FASE 1: uma peca de cada vez ──────────────────────────────────────────
    print("=== FASE 1: Point clouds das pecas ===\n")
    piece_clouds = {}

    for piece_name, info in PIECES.items():
        cx, cy = info["center"]
        sx, sy = info["size"]

        # Mostra só esta peca
        show_only(piece_name)
        print(f"[{piece_name}] visivel — restantes escondidas")

        set_camera(cx, cy, CAM_Z_PIECE, rot_z=0.0)
        await rep.orchestrator.step_async(rt_subframes=8)

        depth = depth_an.get_data()
        rgb   = rgb_an.get_data()
        d     = np.nan_to_num(depth.astype(np.float32), nan=0, posinf=0, neginf=0)

        # Mesa a ~0.15m da camara
        d_mesa = get_surface_depth(d, 0.10, 0.25)
        if d_mesa is None:
            d_mesa = get_surface_depth(d, 0.05, 0.40)
        if d_mesa is None:
            print(f"  [AVISO] mesa nao detetada")
            continue

        # ROI centrada sobre a peca
        margin    = 2.0
        corners_w = [
            (cx - sx*margin, cy - sy*margin),
            (cx + sx*margin, cy - sy*margin),
            (cx - sx*margin, cy + sy*margin),
            (cx + sx*margin, cy + sy*margin),
        ]
        px_c   = [world_to_pixel(wx,wy,cx,cy,CAM_Z_PIECE) for wx,wy in corners_w]
        roi_x1 = max(0,   min(p[0] for p in px_c))
        roi_x2 = min(640, max(p[0] for p in px_c))
        roi_y1 = max(0,   min(p[1] for p in px_c))
        roi_y2 = min(480, max(p[1] for p in px_c))

        roi_d      = d[roi_y1:roi_y2, roi_x1:roi_x2]
        piece_roi  = (roi_d > 0.05) & (roi_d < d_mesa - 0.003)
        piece_mask = np.zeros(d.shape, dtype=bool)
        piece_mask[roi_y1:roi_y2, roi_x1:roi_x2] = piece_roi

        print(f"  cam_z={CAM_Z_PIECE}m  d_mesa={d_mesa:.4f}m  pixels={piece_mask.sum()}")

        if piece_mask.sum() < 20:
            print(f"  [AVISO] poucos pixels — verifica posicao")
            continue

        pc = build_pointcloud(d, piece_mask, cx, cy, CAM_Z_PIECE,
                              d_ref=d_mesa, global_scale=GLOBAL_SCALE)
        if pc is None:
            continue

        piece_clouds[piece_name] = pc
        np.save(str(OUT / f"pc_{piece_name}.npy"), pc)

        debug = rgb[:,:,:3].copy()
        debug[piece_mask] = (debug[piece_mask]*0.3
                             + np.array([255,60,60])*0.7).astype(np.uint8)
        cv2.rectangle(debug,(roi_x1,roi_y1),(roi_x2,roi_y2),(0,255,0),1)
        cv2.imwrite(str(OUT / f"debug_{piece_name}.png"),
                    cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))

        print(f"  pc shape={pc.shape}")
        print(f"  x=[{pc[:,0].min():.3f},{pc[:,0].max():.3f}]  "
              f"y=[{pc[:,1].min():.3f},{pc[:,1].max():.3f}]  "
              f"z=[{pc[:,2].min():.3f},{pc[:,2].max():.3f}]")

    # Restaura todas as pecas visiveis
    show_all()
    print("\n[OK] todas as pecas visiveis novamente")

    # ── FASE 2: cavidades ─────────────────────────────────────────────────────
    print("\n=== FASE 2: Point clouds das cavidades ===\n")

    cx, cy = BOARD["center"]
    set_camera(cx, cy, CAM_Z_BOARD, rot_z=-90.0)
    await rep.orchestrator.step_async(rt_subframes=8)

    depth = depth_an.get_data()
    d     = np.nan_to_num(depth.astype(np.float32), nan=0, posinf=0, neginf=0)

    bx1, by1, bx2, by2 = BOARD["bounds"]
    corners    = [(bx1,by1),(bx2,by1),(bx1,by2),(bx2,by2)]
    px_corners = [world_to_pixel(wx,wy,cx,cy,CAM_Z_BOARD) for wx,wy in corners]
    roi_x1 = max(0,   min(p[0] for p in px_corners) - 5)
    roi_x2 = min(640, max(p[0] for p in px_corners) + 5)
    roi_y1 = max(0,   min(p[1] for p in px_corners) - 5)
    roi_y2 = min(480, max(p[1] for p in px_corners) + 5)

    roi_d        = d[roi_y1:roi_y2, roi_x1:roi_x2]
    d_board_surf = get_surface_depth(roi_d, 0.50, 0.70)
    print(f"d_board_surface={d_board_surf:.4f}m")

    hole_mask_roi = (
            (roi_d > d_board_surf + 0.001) &
            (roi_d < d_board_surf + 0.05)  &
            (roi_d > 0.05)
    )
    hole_mask = np.zeros(d.shape, dtype=bool)
    hole_mask[roi_y1:roi_y2, roi_x1:roi_x2] = hole_mask_roi

    n_h, h_labels, h_stats, h_centroids = cv2.connectedComponentsWithStats(
        hole_mask.astype(np.uint8) * 255)

    blobs = sorted(
        [(int(h_centroids[i][0]), int(h_centroids[i][1]),
          h_stats[i, cv2.CC_STAT_AREA], i)
         for i in range(1, n_h)
         if h_stats[i, cv2.CC_STAT_AREA] >= 50],
        key=lambda b: b[2], reverse=True
    )

    print(f"buracos encontrados: {len(blobs)}\n")
    cavity_clouds = {}

    for idx, (bcx, bcy, area, label_idx) in enumerate(blobs):
        blob_mask = np.zeros(d.shape, dtype=bool)
        blob_mask[h_labels == label_idx] = True

        ys, xs = np.where(blob_mask)
        if len(xs) == 0:
            continue

        wx_top, wy_top = pixels_to_world(xs.astype(float), ys.astype(float),
                                         cx, cy, CAM_Z_BOARD)
        wz_top  = d[ys, xs] - d_board_surf
        top_pts = np.stack([wx_top, wy_top, wz_top], axis=1)
        cx_obj  = wx_top.mean()
        cy_obj  = wy_top.mean()
        z_max   = max(wz_top.max(), 0.001)

        wall_list = []
        contours, _ = cv2.findContours(blob_mask.astype(np.uint8) * 255,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if contours:
            main_c = max(contours, key=len)
            cpts   = main_c.reshape(-1, 2)
            cwx, cwy = pixels_to_world(cpts[:,0].astype(float),
                                       cpts[:,1].astype(float),
                                       cx, cy, CAM_Z_BOARD)
            for level in range(9):
                wall_z   = z_max * level / 8
                wall_pts = np.stack(
                    [cwx, cwy, np.full(len(cwx), wall_z)], axis=1)
                wall_list.append(wall_pts)

        points = np.vstack([top_pts] + wall_list).astype(np.float32)
        points[:, 0] -= cx_obj
        points[:, 1] -= cy_obj
        points[:, 2] -= points[:, 2].min()
        points        /= GLOBAL_SCALE

        replace = len(points) < N_POINTS
        idx_s   = np.random.choice(len(points), N_POINTS, replace=replace)
        pc      = points[idx_s]

        cavity_clouds[idx] = pc
        np.save(str(OUT / f"pc_cavity_{idx}.npy"), pc)

        print(f"  cavidade {idx}: area={area}px")
        print(f"    x=[{pc[:,0].min():.3f},{pc[:,0].max():.3f}]  "
              f"y=[{pc[:,1].min():.3f},{pc[:,1].max():.3f}]  "
              f"z=[{pc[:,2].min():.3f},{pc[:,2].max():.3f}]")

    print(f"\n=== Sumario ===")
    print(f"  Pecas:     {list(piece_clouds.keys())}")
    print(f"  Cavidades: {len(cavity_clouds)}")

    return piece_clouds, cavity_clouds

asyncio.ensure_future(generate_pointclouds_final())