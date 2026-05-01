import asyncio
import math

async def diagnose_cav_depth():
    import omni.replicator.core as rep
    import numpy as np
    from pxr import UsdGeom, Gf
    import omni.usd

    BOARD_CAM_Z  = 1.00
    CAV_CAM_Z    = 0.46
    BOARD_CENTER = (0.2885, 0.0020)

    CAV_PIXELS = {0:(332,227), 1:(332,257), 2:(306,257), 3:(306,224)}
    CAV_NAMES  = {0:"rectangle", 1:"square", 2:"circle", 3:"star"}
    CAV_SIZE   = {0:(0.0315,0.0180), 1:(0.0180,0.0180),
                  2:(0.0180,0.0179), 3:(0.0171,0.0163)}

    def set_camera(x, y, z):
        stage    = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath("/World/Camera")
        ops_dict = {op.GetOpName(): op
                    for op in UsdGeom.Xformable(cam_prim).GetOrderedXformOps()}
        ops_dict["xformOp:translate"].Set(Gf.Vec3d(x, y, z))
        if "xformOp:orient" in ops_dict:
            ops_dict["xformOp:orient"].Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    def pixel_to_world(u, v, cam_x, cam_y, cam_z, img_w=640, img_h=480,
                       focal_mm=24.0, aperture_mm=36.0):
        fov_h = 2 * math.atan((aperture_mm/2)/focal_mm)
        fov_v = fov_h * (img_h/img_w)
        mpp_x = (2*cam_z*math.tan(fov_h/2))/img_w
        mpp_y = (2*cam_z*math.tan(fov_v/2))/img_h
        return cam_x+(u-img_w/2)*mpp_x, cam_y-(v-img_h/2)*mpp_y

    def world_to_pixel(wx, wy, cam_x, cam_y, cam_z, img_w=640, img_h=480,
                       focal_mm=24.0, aperture_mm=36.0):
        fov_h = 2 * math.atan((aperture_mm/2)/focal_mm)
        fov_v = fov_h * (img_h/img_w)
        mpp_x = (2*cam_z*math.tan(fov_h/2))/img_w
        mpp_y = (2*cam_z*math.tan(fov_v/2))/img_h
        return int(img_w/2+(wx-cam_x)/mpp_x), int(img_h/2-(wy-cam_y)/mpp_y)

    cam_bx, cam_by = BOARD_CENTER
    cav_world = {i: pixel_to_world(pu,pv,cam_bx,cam_by,BOARD_CAM_Z)
                 for i,(pu,pv) in CAV_PIXELS.items()}

    rp       = rep.create.render_product("/World/Camera", (640,480))
    depth_an = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_an.attach([rp])

    for cav_idx,(cav_wx,cav_wy) in cav_world.items():
        name   = CAV_NAMES[cav_idx]
        sx, sy = CAV_SIZE[cav_idx]

        set_camera(cav_wx, cav_wy, CAV_CAM_Z)
        await rep.orchestrator.step_async(rt_subframes=8)

        depth = depth_an.get_data()
        d     = np.nan_to_num(depth.astype(np.float32), nan=0, posinf=0, neginf=0)

        # ROI com margem 3x
        margin = 3.0
        corners = [(cav_wx-sx*margin, cav_wy-sy*margin),
                   (cav_wx+sx*margin, cav_wy+sy*margin)]
        px = [world_to_pixel(wx,wy,cav_wx,cav_wy,CAV_CAM_Z) for wx,wy in corners]
        rx1 = max(0,   min(p[0] for p in px))
        rx2 = min(640, max(p[0] for p in px))
        ry1 = max(0,   min(p[1] for p in px))
        ry2 = min(480, max(p[1] for p in px))

        roi_d = d[ry1:ry2, rx1:rx2]
        valid = roi_d[(roi_d > 0.01) & (roi_d < 0.12)]

        print(f"\n[cav_{cav_idx} — {name}]  roi={rx1}:{rx2},{ry1}:{ry2}")
        print(f"  depth range na ROI: {valid.min():.4f} – {valid.max():.4f}m" if len(valid) > 0 else "  ROI vazia")
        print(f"  Histograma (passo 1mm):")
        hist, edges = np.histogram(valid, bins=np.arange(0.01, 0.12, 0.001))
        for i,count in enumerate(hist):
            if count > 0:
                bar = "█" * min(20, count//10)
                print(f"    {edges[i]:.3f}m: {bar} ({count})")

asyncio.ensure_future(diagnose_cav_depth())