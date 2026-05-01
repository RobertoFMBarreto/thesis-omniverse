import asyncio
import math
import numpy as np

async def detect_cavities_final():
    import omni.usd
    import omni.replicator.core as rep
    import cv2
    from pathlib import Path
    from pxr import UsdGeom, Gf

    OUT = Path("/tmp/shape_insertion/data/raw")

    CAM_X, CAM_Y, CAM_Z = 0.2885, 0.0020, 0.60
    CAM_ROT = -90.0
    N_POINTS     = 1024
    GLOBAL_SCALE = 0.05
    CAV_NAMES    = {0:"rectangle", 1:"square", 2:"circle", 3:"star"}

    stage    = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/Camera")
    ops_dict = {op.GetOpName(): op
                for op in UsdGeom.Xformable(cam_prim).GetOrderedXformOps()}
    ops_dict["xformOp:translate"].Set(Gf.Vec3d(CAM_X, CAM_Y, CAM_Z))
    half = math.radians(CAM_ROT)/2
    ops_dict["xformOp:orient"].Set(
        Gf.Quatd(math.cos(half), 0.0, 0.0, math.sin(half)))

    rp       = rep.create.render_product("/World/Camera",(640,480))
    depth_an = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_an   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_an.attach([rp]); rgb_an.attach([rp])
    await rep.orchestrator.step_async(rt_subframes=8)

    depth = depth_an.get_data()
    rgb   = rgb_an.get_data()
    d     = np.nan_to_num(depth.astype(np.float32),nan=0,posinf=0,neginf=0)

    # Surface da board
    zone = d[(d>0.15)&(d<0.25)]
    hist,edges = np.histogram(zone, bins=np.arange(0.15,0.25,0.001))
    d_board = edges[np.argmax(hist)]+0.0005
    print(f"d_board_surface={d_board:.4f}m")

    # Buracos = mais perto que a surface
    hole_mask = (d > 0.15) & (d < d_board - 0.005) & (d < 0.25)
    print(f"pixels buraco: {hole_mask.sum()}")

    n,labels,stats,cents = cv2.connectedComponentsWithStats(
        hole_mask.astype(np.uint8)*255)
    blobs = sorted([(int(cents[i][0]),int(cents[i][1]),
                     stats[i,cv2.CC_STAT_AREA],i)
                    for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=30],
                   key=lambda b:b[2],reverse=True)

    print(f"buracos detetados: {len(blobs)}\n")

    # Converte pixel→world com rot=-90°
    focal_mm, aperture = 24.0, 36.0
    img_w, img_h = 640, 480
    fov_h = 2*math.atan((aperture/2)/focal_mm)
    fov_v = fov_h*(img_h/img_w)
    mpp_x = (2*CAM_Z*math.tan(fov_h/2))/img_w
    mpp_y = (2*CAM_Z*math.tan(fov_v/2))/img_h

    debug = rgb[:,:,:3].copy()
    debug[hole_mask]=(debug[hole_mask]*0.3
                      +np.array([60,60,255])*0.7).astype(np.uint8)

    cav_world = {}
    for idx,(bcx,bcy,area,_) in enumerate(blobs[:4]):
        du = bcx - img_w/2
        dv = bcy - img_h/2
        # rot=-90°: du→-Y world, dv→-X world
        wx = CAM_X - dv * mpp_y
        wy = CAM_Y - du * mpp_x
        cav_world[idx] = (wx, wy)
        print(f"  buraco {idx}: pixel=({bcx},{bcy})  "
              f"area={area}px  world=({wx:.4f},{wy:.4f})")
        cv2.circle(debug,(bcx,bcy),6,(255,255,0),-1)
        cv2.putText(debug,f"{idx}",
                    (bcx+8,bcy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)

    cv2.imwrite(str(OUT/"debug_holes_detected.png"),
                cv2.cvtColor(debug,cv2.COLOR_RGB2BGR))
    print(f"\ndocker cp isaac-sim-v2:{OUT}/debug_holes_detected.png ./")

    # Passo 2: captura hires de cada cavidade
    print(f"\n=== Captura hires ===\n")

    def set_camera(x, y, z, rot_z=0.0):
        ops_dict["xformOp:translate"].Set(Gf.Vec3d(x, y, z))
        half = math.radians(rot_z)/2
        ops_dict["xformOp:orient"].Set(
            Gf.Quatd(math.cos(half),0.0,0.0,math.sin(half)))

    def get_surface_depth(d, d_min, d_max):
        zone = d[(d>d_min)&(d<d_max)]
        if len(zone)==0: return None
        hist,edges = np.histogram(zone,bins=np.arange(d_min,d_max,0.001))
        return edges[np.argmax(hist)]+0.0005

    def build_pointcloud(d, mask, cam_x, cam_y, cam_z,
                         d_ref, global_scale, n_points=1024):
        ys,xs = np.where(mask)
        if len(xs)==0: return None
        fov_h = 2*math.atan((36.0/2)/24.0)
        fov_v = fov_h*(480/640)
        mpp_x = (2*cam_z*math.tan(fov_h/2))/640
        mpp_y = (2*cam_z*math.tan(fov_v/2))/480
        wx_top = cam_x + (xs-320)*mpp_x
        wy_top = cam_y - (ys-240)*mpp_y
        wz_top = d_ref - d[ys,xs]
        top_pts= np.stack([wx_top,wy_top,wz_top],axis=1)
        cx_obj = wx_top.mean(); cy_obj = wy_top.mean()
        z_max  = max(wz_top.max(),0.001)
        wall_list=[]
        contours,_=cv2.findContours(mask.astype(np.uint8)*255,
                                    cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if contours:
            mc=max(contours,key=len); cpts=mc.reshape(-1,2)
            cwx=cam_x+(cpts[:,0]-320)*mpp_x
            cwy=cam_y-(cpts[:,1]-240)*mpp_y
            for lv in range(9):
                wz=z_max*lv/8
                wall_list.append(np.stack([cwx,cwy,np.full(len(cwx),wz)],axis=1))
        points=np.vstack([top_pts]+wall_list).astype(np.float32)
        points[:,0]-=cx_obj; points[:,1]-=cy_obj
        points[:,2]-=points[:,2].min()
        points/=global_scale
        replace=len(points)<n_points
        return points[np.random.choice(len(points),n_points,replace=replace)]

    CAM_CAV_Z = 0.46

    for cav_idx,(cav_wx,cav_wy) in cav_world.items():
        set_camera(cav_wx, cav_wy, CAM_CAV_Z, rot_z=0.0)
        await rep.orchestrator.step_async(rt_subframes=8)
        depth=depth_an.get_data()
        rgb=rgb_an.get_data()
        d=np.nan_to_num(depth.astype(np.float32),nan=0,posinf=0,neginf=0)

        d_surf = get_surface_depth(d, 0.01, 0.08)
        if d_surf is None:
            print(f"[cav_{cav_idx}] surface nao detetada"); continue

        # Buracos mais perto que a surface
        fov_h2=2*math.atan((36.0/2)/24.0)
        fov_v2=fov_h2*(480/640)
        mpp_xc=(2*CAM_CAV_Z*math.tan(fov_h2/2))/640
        mpp_yc=(2*CAM_CAV_Z*math.tan(fov_v2/2))/480
        margin=6.0
        sx,sy=0.020,0.020
        hu=int((sx*margin)/mpp_xc); hv=int((sy*margin)/mpp_yc)
        rx1=max(0,320-hu); rx2=min(640,320+hu)
        ry1=max(0,240-hv); ry2=min(480,240+hv)

        roi_d=d[ry1:ry2,rx1:rx2]
        hole_roi=(roi_d>0.01)&(roi_d<d_surf-0.003)
        hole_mask=np.zeros(d.shape,dtype=bool)
        hole_mask[ry1:ry2,rx1:rx2]=hole_roi

        # Componente mais central
        n_c,c_labels,c_stats,c_cents=cv2.connectedComponentsWithStats(
            hole_mask.astype(np.uint8)*255)
        best=1; best_dist=float('inf')
        for i in range(1,n_c):
            if c_stats[i,cv2.CC_STAT_AREA]<20: continue
            dist=math.sqrt((c_cents[i][0]-320)**2+(c_cents[i][1]-240)**2)
            if dist<best_dist:
                best_dist=dist; best=i
        hole_final=np.zeros(d.shape,dtype=bool)
        if n_c>=2:
            hole_final[c_labels==best]=True

        print(f"[cav_{cav_idx}] d_surf={d_surf:.4f}m  pixels={hole_final.sum()}")

        debug2=rgb[:,:,:3].copy()
        debug2[hole_final]=(debug2[hole_final]*0.3
                            +np.array([60,60,255])*0.7).astype(np.uint8)
        cv2.rectangle(debug2,(rx1,ry1),(rx2,ry2),(0,255,0),1)
        cv2.circle(debug2,(320,240),4,(255,255,0),-1)
        cv2.imwrite(str(OUT/f"debug_cav_{cav_idx}_hires.png"),
                    cv2.cvtColor(debug2,cv2.COLOR_RGB2BGR))

        if hole_final.sum()<20:
            print(f"  [AVISO] buraco nao detetado\n"); continue

        pc=build_pointcloud(d,hole_final,cav_wx,cav_wy,CAM_CAV_Z,
                            d_ref=d_surf,global_scale=GLOBAL_SCALE)
        if pc is None:
            print(f"  [AVISO] pc vazia\n"); continue

        np.save(str(OUT/f"pc_cavity_{cav_idx}_hires.npy"),pc)
        print(f"  pc={pc.shape}  "
              f"x=[{pc[:,0].min():.3f},{pc[:,0].max():.3f}]  "
              f"y=[{pc[:,1].min():.3f},{pc[:,1].max():.3f}]  "
              f"z=[{pc[:,2].min():.3f},{pc[:,2].max():.3f}]\n")

    print("=== Concluido ===")
    for i in range(len(cav_world)):
        print(f"  docker cp isaac-sim-v2:{OUT}/debug_cav_{i}_hires.png ./")

asyncio.ensure_future(detect_cavities_final())