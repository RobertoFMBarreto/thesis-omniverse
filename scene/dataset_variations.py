import asyncio
import math
import numpy as np

async def generate_dataset_variations():
    import omni.usd
    import omni.replicator.core as rep
    import cv2
    from pathlib import Path
    from pxr import UsdGeom, Gf

    OUT = Path("/tmp/shape_insertion/data/variations")
    OUT.mkdir(parents=True, exist_ok=True)

    N_POINTS      = 1024
    GLOBAL_SCALE  = 0.05
    CAM_Z_PIECE   = 0.58
    N_VARIATIONS  = 10   # posicoes por peca

    # Bounds originais de cada peca
    PIECES = {
        "rectangle": {"center": (-0.2552, 0.4478), "size": (0.0315, 0.0180)},
        "star":      {"center": (-0.3528, 0.4186), "size": (0.0171, 0.0163)},
        "square":    {"center": (-0.1468, 0.4119), "size": (0.0180, 0.0180)},
        "circle":    {"center": (-0.3121, 0.3876), "size": (0.0180, 0.0179)},
    }

    # Area da mesa de pecas onde podemos mover as pecas
    # Mesa: position=(-0.25, 0.45) scale=(0.70, 0.20)
    # Area segura (evita bordas): x in [-0.50, 0.00], y in [0.37, 0.53]
    MESA_X_RANGE = (-0.50, 0.00)
    MESA_Y_RANGE = ( 0.37, 0.53)

    rng = np.random.default_rng(42)

    def set_camera(x, y, z, rot_z=0.0):
        stage    = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath("/World/Camera")
        ops_dict = {op.GetOpName(): op
                    for op in UsdGeom.Xformable(cam_prim).GetOrderedXformOps()}
        ops_dict["xformOp:translate"].Set(Gf.Vec3d(x, y, z))
        half = math.radians(rot_z) / 2
        quat = Gf.Quatd(math.cos(half), 0.0, 0.0, math.sin(half))
        if "xformOp:orient" in ops_dict:
            ops_dict["xformOp:orient"].Set(quat)

    def move_piece(piece_name, new_x, new_y):
        """Move a peca para uma nova posicao XY mantendo Z."""
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(f"/World/{piece_name}")
        if not prim.IsValid():
            return False
        ops_dict = {op.GetOpName(): op
                    for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        if "xformOp:translate" in ops_dict:
            current = ops_dict["xformOp:translate"].Get()
            ops_dict["xformOp:translate"].Set(Gf.Vec3d(new_x, new_y, current[2]))
            return True
        return False

    def show_only(piece_name):
        stage = omni.usd.get_context().get_stage()
        for name in PIECES.keys():
            prim = stage.GetPrimAtPath(f"/World/{name}")
            if prim.IsValid():
                img = UsdGeom.Imageable(prim)
                img.MakeVisible() if name == piece_name else img.MakeInvisible()

    def show_all():
        stage = omni.usd.get_context().get_stage()
        for name in PIECES.keys():
            prim = stage.GetPrimAtPath(f"/World/{name}")
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeVisible()

    def get_surface_depth(d, d_min, d_max):
        zone = d[(d > d_min) & (d < d_max)]
        if len(zone) == 0:
            return None
        hist, edges = np.histogram(zone, bins=np.arange(d_min, d_max, 0.001))
        return edges[np.argmax(hist)] + 0.0005

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

    print("=== Geracao de variacoes de posicao ===\n")
    total_saved = 0

    for piece_name, info in PIECES.items():
        cx_orig, cy_orig = info["center"]
        sx, sy           = info["size"]

        print(f"[{piece_name}] a gerar {N_VARIATIONS} variacoes...")

        show_only(piece_name)

        saved_this_piece = 0
        attempts         = 0

        while saved_this_piece < N_VARIATIONS and attempts < N_VARIATIONS * 3:
            attempts += 1

            # Posicao aleatoria dentro da mesa
            new_x = rng.uniform(MESA_X_RANGE[0], MESA_X_RANGE[1])
            new_y = rng.uniform(MESA_Y_RANGE[0], MESA_Y_RANGE[1])

            # Move a peca
            if not move_piece(piece_name, new_x, new_y):
                print(f"  [AVISO] nao consegui mover {piece_name}")
                break

            # Camara centrada sobre a nova posicao
            set_camera(new_x, new_y, CAM_Z_PIECE, rot_z=0.0)
            await rep.orchestrator.step_async(rt_subframes=8)

            depth = depth_an.get_data()
            d     = np.nan_to_num(depth.astype(np.float32), nan=0,
                                  posinf=0, neginf=0)

            d_mesa = get_surface_depth(d, 0.10, 0.25)
            if d_mesa is None:
                continue

            # ROI centrada sobre a nova posicao
            margin    = 2.0
            corners_w = [
                (new_x - sx*margin, new_y - sy*margin),
                (new_x + sx*margin, new_y - sy*margin),
                (new_x - sx*margin, new_y + sy*margin),
                (new_x + sx*margin, new_y + sy*margin),
            ]
            px_c   = [world_to_pixel(wx,wy,new_x,new_y,CAM_Z_PIECE)
                      for wx,wy in corners_w]
            roi_x1 = max(0,   min(p[0] for p in px_c))
            roi_x2 = min(640, max(p[0] for p in px_c))
            roi_y1 = max(0,   min(p[1] for p in px_c))
            roi_y2 = min(480, max(p[1] for p in px_c))

            roi_d      = d[roi_y1:roi_y2, roi_x1:roi_x2]
            piece_roi  = (roi_d > 0.05) & (roi_d < d_mesa - 0.003)
            piece_mask = np.zeros(d.shape, dtype=bool)
            piece_mask[roi_y1:roi_y2, roi_x1:roi_x2] = piece_roi

            if piece_mask.sum() < 20:
                continue

            pc = build_pointcloud(d, piece_mask, new_x, new_y, CAM_Z_PIECE,
                                  d_ref=d_mesa, global_scale=GLOBAL_SCALE)
            if pc is None:
                continue

            # Guarda com nome que inclui variacao
            fname = f"pc_{piece_name}_v{saved_this_piece:02d}.npy"
            np.save(str(OUT / fname), pc)
            saved_this_piece += 1
            total_saved      += 1
            print(f"  v{saved_this_piece-1:02d}: pos=({new_x:.3f},{new_y:.3f})  "
                  f"pixels={piece_mask.sum()}  guardado")

        # Restaura posicao original
        move_piece(piece_name, cx_orig, cy_orig)
        print(f"  [{piece_name}] {saved_this_piece} variacoes guardadas\n")

    show_all()

    print(f"=== Sumario ===")
    print(f"  Total point clouds guardadas: {total_saved}")
    print(f"  Em: {OUT}")

asyncio.ensure_future(generate_dataset_variations())