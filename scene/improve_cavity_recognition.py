import asyncio
import math

async def capture_cavities_hires_v5():
    import omni.usd
    import omni.replicator.core as rep
    import numpy as np
    import cv2
    from pathlib import Path
    from pxr import UsdGeom, Gf

    OUT = Path("/tmp/shape_insertion/data/raw")
    OUT.mkdir(parents=True, exist_ok=True)

    N_POINTS     = 1024
    GLOBAL_SCALE = 0.05
    CAV_CAM_Z    = 0.46
    BOARD_CAM_Z  = 1.00
    BOARD_CENTER = (0.2885, 0.0020)

    BOARD_SURFACE   = 0.0395
    HOLE_MIN_OFFSET = 0.001
    HOLE_MAX_OFFSET = 0.019

    # Tamanho estimado das cavidades em metros
    # Para calcular a ROI em pixels
    CAV_SIZE = {
        0: (0.0315, 0.0180),   # rectangle
        1: (0.0180, 0.0180),   # square
        2: (0.0180, 0.0179),   # circle
        3: (0.0171, 0.0163),   # star
    }

    CAV_PIXELS = {
        0: (332, 227),
        1: (332, 257),
        2: (306, 257),
        3: (306, 224),
    }
    CAV_NAMES = {0: "rectangle", 1: "square", 2: "circle", 3: "star"}

    def set_camera(x, y, z):
        stage    = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath("/World/Camera")
        ops_dict = {op.GetOpName(): op
                    for op in UsdGeom.Xformable(cam_prim).GetOrderedXformOps()}
        ops_dict["xformOp:translate"].Set(Gf.Vec3d(x, y, z))
        if "xformOp:orient" in ops_dict:
            ops_dict["xformOp:orient"].Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    def pixel_to_world(u, v, cam_x, cam_y, cam_z,
                       img_w=640, img_h=480,
                       focal_mm=24.0, aperture_mm=36.0):
        fov_h = 2 * math.atan((aperture_mm / 2) / focal_mm)
        fov_v = fov_h * (img_h / img_w)
        mpp_x = (2 * cam_z * math.tan(fov_h / 2)) / img_w
        mpp_y = (2 * cam_z * math.tan(fov_v / 2)) / img_h
        wx = cam_x + (u - img_w/2) * mpp_x
        wy = cam_y - (v - img_h/2) * mpp_y
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

    def pixels_to_world_arr(us, vs, cam_x, cam_y, cam_z,
                            img_w=640, img_h=480,
                            focal_mm=24.0, aperture_mm=36.0):
        fov_h = 2 * math.atan((aperture_mm / 2) / focal_mm)
        fov_v = fov_h * (img_h / img_w)
        mpp_x = (2 * cam_z * math.tan(fov_h / 2)) / img_w
        mpp_y = (2 * cam_z * math.tan(fov_v / 2)) / img_h
        wx = cam_x + (us - img_w/2) * mpp_x
        wy = cam_y - (vs - img_h/2) * mpp_y
        return wx, wy

    def build_pointcloud(d, mask, cam_x, cam_y, cam_z,
                         d_ref, global_scale, n_points=1024):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None

        wx_top, wy_top = pixels_to_world_arr(xs.astype(float), ys.astype(float),
                                             cam_x, cam_y, cam_z)
        wz_top  = d[ys, xs] - d_ref
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
            cwx, cwy = pixels_to_world_arr(cpts[:,0].astype(float),
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

    # Calcula posicoes world das cavidades a partir dos pixels da board cam
    cam_bx, cam_by = BOARD_CENTER
    cav_world = {}
    for cav_idx, (pu, pv) in CAV_PIXELS.items():
        wx, wy = pixel_to_world(pu, pv, cam_bx, cam_by, BOARD_CAM_Z)
        cav_world[cav_idx] = (wx, wy)

    rp       = rep.create.render_product("/World/Camera", (640, 480))
    depth_an = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_an   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_an.attach([rp])
    rgb_an.attach([rp])

    print(f"=== Captura hires v5 — com ROI ===\n")

    for cav_idx, (cav_wx, cav_wy) in cav_world.items():
        name   = CAV_NAMES[cav_idx]
        sx, sy = CAV_SIZE[cav_idx]

        # Camara centrada sobre a cavidade
        set_camera(cav_wx, cav_wy, CAV_CAM_Z)
        await rep.orchestrator.step_async(rt_subframes=8)

        depth = depth_an.get_data()
        rgb   = rgb_an.get_data()
        d     = np.nan_to_num(depth.astype(np.float32), nan=0, posinf=0, neginf=0)

        # ROI em pixels — cavidade deve estar no centro da imagem
        # cam esta centrada sobre a cavidade → cavidade aparece em (320, 240)
        # ROI: tamanho da cavidade + margem 3x
        margin = 3.0
        corners_w = [
            (cav_wx - sx*margin, cav_wy - sy*margin),
            (cav_wx + sx*margin, cav_wy - sy*margin),
            (cav_wx - sx*margin, cav_wy + sy*margin),
            (cav_wx + sx*margin, cav_wy + sy*margin),
        ]
        px_c   = [world_to_pixel(wx, wy, cav_wx, cav_wy, CAV_CAM_Z)
                  for wx, wy in corners_w]
        roi_x1 = max(0,   min(p[0] for p in px_c))
        roi_x2 = min(640, max(p[0] for p in px_c))
        roi_y1 = max(0,   min(p[1] for p in px_c))
        roi_y2 = min(480, max(p[1] for p in px_c))

        # Threshold dentro da ROI
        roi_d = d[roi_y1:roi_y2, roi_x1:roi_x2]
        hole_roi = (
                (roi_d > BOARD_SURFACE + HOLE_MIN_OFFSET) &
                (roi_d < BOARD_SURFACE + HOLE_MAX_OFFSET)
        )
        hole_mask = np.zeros(d.shape, dtype=bool)
        hole_mask[roi_y1:roi_y2, roi_x1:roi_x2] = hole_roi

        print(f"[cav_{cav_idx} — {name}]")
        print(f"  cam: ({cav_wx:.4f}, {cav_wy:.4f}, {CAV_CAM_Z})")
        print(f"  roi: x=[{roi_x1}:{roi_x2}]  y=[{roi_y1}:{roi_y2}]  "
              f"size={roi_x2-roi_x1}×{roi_y2-roi_y1}px")
        print(f"  hole_mask pixels: {hole_mask.sum()}")

        # Debug visual
        debug = rgb[:,:,:3].copy()
        debug[hole_mask] = (debug[hole_mask]*0.3
                            + np.array([60,60,255])*0.7).astype(np.uint8)
        cv2.rectangle(debug, (roi_x1,roi_y1), (roi_x2,roi_y2), (0,255,0), 1)
        cv2.imwrite(str(OUT / f"debug_cav_{cav_idx}_hires.png"),
                    cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))

        if hole_mask.sum() < 20:
            print(f"  [AVISO] buraco nao detetado\n")
            continue

        pc = build_pointcloud(d, hole_mask, cav_wx, cav_wy, CAV_CAM_Z,
                              d_ref=BOARD_SURFACE, global_scale=GLOBAL_SCALE)
        if pc is None:
            print(f"  [AVISO] point cloud vazia\n")
            continue

        np.save(str(OUT / f"pc_cavity_{cav_idx}_hires.npy"), pc)

        print(f"  pc: {pc.shape}")
        print(f"  x=[{pc[:,0].min():.3f},{pc[:,0].max():.3f}]  "
              f"y=[{pc[:,1].min():.3f},{pc[:,1].max():.3f}]  "
              f"z=[{pc[:,2].min():.3f},{pc[:,2].max():.3f}]")
        print(f"  guardado: pc_cavity_{cav_idx}_hires.npy\n")

    print("=== Concluido ===")
    for i in range(4):
        print(f"  docker cp isaac-sim-v2:{OUT}/debug_cav_{i}_hires.png ./")

asyncio.ensure_future(capture_cavities_hires_v5())