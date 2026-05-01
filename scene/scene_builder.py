import asyncio

async def rebuild_scene_final():
    import omni.usd
    import omni.kit.commands
    import numpy as np
    from pxr import UsdGeom, Gf
    from omni.isaac.core.objects import VisualCuboid

    stage = omni.usd.get_context().get_stage()

    for p in (["/World/TablePieces", "/World/TableCavities",
               "/World/Piece", "/World/Board", "/World/Camera"]
              + [f"/World/Cavity_{i}" for i in range(6)]):
        if stage.GetPrimAtPath(p).IsValid():
            omni.kit.commands.execute("DeletePrimsCommand", paths=[p])

    OBJ_H        = 0.05
    TABLE_H      = 0.40   # altura de ambas as mesas
    TABLE_TOP    = TABLE_H
    CAM_HEIGHT   = TABLE_TOP + 0.60   # câmara 60cm acima da superfície das mesas

    # ── Mesa de peças ─────────────────────────────────────────────────────────
    VisualCuboid(
        prim_path="/World/TablePieces",
        position=np.array([-0.25, 0.45, TABLE_H / 2]),
        scale=np.array([0.70, 0.20, TABLE_H]),
        color=np.array([0.6, 0.5, 0.35]),
    )

    # Peça centrada na mesa de peças
    VisualCuboid(
        prim_path="/World/Piece",
        position=np.array([-0.25, 0.45, TABLE_TOP + OBJ_H / 2]),
        scale=np.array([0.06, 0.06, OBJ_H]),
        color=np.array([0.9, 0.2, 0.2]),
    )

    # ── Mesa de cavidades ─────────────────────────────────────────────────────
    VisualCuboid(
        prim_path="/World/TableCavities",
        position=np.array([0.30, 0.0, TABLE_H / 2]),
        scale=np.array([0.25, 0.45, TABLE_H]),
        color=np.array([0.5, 0.55, 0.6]),
    )

    # Board
    VisualCuboid(
        prim_path="/World/Board",
        position=np.array([0.30, 0.0, TABLE_TOP + OBJ_H / 2]),
        scale=np.array([0.20, 0.38, OBJ_H]),
        color=np.array([0.85, 0.82, 0.75]),
    )

    # Cavidades
    cavities = [
        (0.23,  0.12, 0.06, 0.06),
        (0.35,  0.12, 0.08, 0.04),
        (0.23, -0.05, 0.08, 0.04),
        (0.35, -0.05, 0.06, 0.06),
    ]
    for i, (cx, cy, sw, sd) in enumerate(cavities):
        VisualCuboid(
            prim_path=f"/World/Cavity_{i}",
            position=np.array([cx, cy, TABLE_TOP + OBJ_H + 0.003]),
            scale=np.array([sw, sd, 0.004]),
            color=np.array([0.3, 0.5, 0.9]),
        )

    # ── Câmara — começa sobre a mesa de peças ────────────────────────────────
    omni.kit.commands.execute("CreatePrimWithDefaultXformCommand",
                              prim_type="Camera", prim_path="/World/Camera")
    cam_prim = stage.GetPrimAtPath("/World/Camera")
    UsdGeom.XformCommonAPI(cam_prim).SetTranslate(
        Gf.Vec3d(-0.25, 0.45, CAM_HEIGHT)   # centrada sobre mesa de peças
    )
    cam = UsdGeom.Camera(cam_prim)
    cam.GetFocalLengthAttr().Set(24.0)
    cam.GetHorizontalApertureAttr().Set(36.0)

    print("[build] cena final pronta")
    print(f"  TABLE_TOP:   {TABLE_TOP:.2f}m")
    print(f"  CAM_HEIGHT:  {CAM_HEIGHT:.2f}m  ({CAM_HEIGHT - TABLE_TOP:.2f}m acima das mesas)")
    print(f"  OBJECT_THRESH: d < {CAM_HEIGHT - TABLE_TOP - 0.008:.3f}m")
    print()
    print("  Posicoes da camara:")
    print(f"    Fase 1 — mesa pecas:     (-0.25, 0.45, {CAM_HEIGHT:.2f})")
    print(f"    Fase 2 — mesa cavidades: ( 0.30, 0.00, {CAM_HEIGHT:.2f})")
    print()
    print("Carrega Play ▶ e corre:")
    print("  capture_piece()     — captura crop da peca")
    print("  capture_cavities()  — captura crop de cada cavidade")

asyncio.ensure_future(rebuild_scene_final())