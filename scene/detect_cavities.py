import asyncio

async def check_and_capture():
    import omni.usd
    import omni.replicator.core as rep
    import numpy as np
    import cv2
    from pathlib import Path
    from pxr import UsdGeom, Gf
    import math

    OUT = Path("/tmp/shape_insertion/data/raw")
    stage    = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/Camera")

    # Le posicao actual
    xf_cache = UsdGeom.XformCache()
    t = xf_cache.GetLocalToWorldTransform(cam_prim).ExtractTranslation()
    print(f"Posicao actual: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")

    # Forcas a posicao correcta sobre a board
    ops_dict = {op.GetOpName(): op
                for op in UsdGeom.Xformable(cam_prim).GetOrderedXformOps()}
    ops_dict["xformOp:translate"].Set(Gf.Vec3d(0.2885, 0.0020, 1.00))
    half = math.radians(-90.0) / 2
    ops_dict["xformOp:orient"].Set(Gf.Quatd(math.cos(half), 0.0, 0.0, math.sin(half)))

    # Confirma
    xf_cache2 = UsdGeom.XformCache()
    t2 = xf_cache2.GetLocalToWorldTransform(cam_prim).ExtractTranslation()
    print(f"Posicao depois:  ({t2[0]:.4f}, {t2[1]:.4f}, {t2[2]:.4f})")

    rp       = rep.create.render_product("/World/Camera", (640,480))
    depth_an = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_an   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_an.attach([rp])
    rgb_an.attach([rp])
    await rep.orchestrator.step_async(rt_subframes=8)

    depth = depth_an.get_data()
    rgb   = rgb_an.get_data()
    d     = np.nan_to_num(depth.astype(np.float32), nan=0, posinf=0, neginf=0)

    print(f"depth range: {d[d>0.1].min():.4f} – {d[d>0.1].max():.4f}m")

    # Histograma
    valid = d[(d>0.3)&(d<1.0)]
    hist, edges = np.histogram(valid, bins=np.arange(0.30,0.80,0.01))
    print("Histograma:")
    for i,count in enumerate(hist):
        if count > 0:
            bar = "█"*min(20,count//500)
            print(f"  {edges[i]:.2f}m: {bar} ({count})")

    cv2.imwrite(str(OUT/"check_board_view.png"),
                cv2.cvtColor(rgb[:,:,:3], cv2.COLOR_RGB2BGR))
    print(f"\ndocker cp isaac-sim-v2:{OUT}/check_board_view.png ./")

asyncio.ensure_future(check_and_capture())