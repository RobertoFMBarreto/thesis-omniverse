# Scene Scale Inspection Report

**Script**: `inspect_cavity_scene_scale.py`
**Timestamp**: 2026-05-09T10:05:54.445062+00:00
**Stage**: `/workspace/Tese_Roberto/shape_insertion/thesis-omniverse/scene/final_stage.usd`

---

## Context

cavity_03 (circular opening) measures **60.84 × 60.84 mm** in perception.
CAD nominal diameter: **51 mm** (clearance-adjusted: 51 mm).
Inflation: **+9.84 mm (+19.3 %)**.
Perception scale is consistent across all 4 cavities (0.749 vs 0.751 mm/px → 0.3 %),
ruling out projection or segmentation errors.  This report investigates the USD scene.

---

## Candidate Prims (8 found)

Sorted by prim path.  Prims with highlights are marked `[!]`.

- [!] `/World/Board_Tese` (Xform)
- [!] `/World/Board_Tese/Body1` (Mesh)
-     `/World/Board_Tese/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C` (Material)
-     `/World/Board_Tese/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C_shader` (Shader)
- [!] `/World/Circle` (Xform)
- [!] `/World/Circle/Body1` (Mesh)
- [!] `/World/Circle/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C` (Material)
- [!] `/World/Circle/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C_shader` (Shader)

---

## Per-Prim Detail

### [!] `/World/Board_Tese`

- **Type**: `Xform`
- **Local scale**: (1.000000, 1.000000, 1.000000)  [unity]
- **BBox available**: `True`
- **BBox method**: `BBoxCache`
- **BBox (world)**: X=0.20000 m (200.00 mm)  Y=0.17500 m (175.00 mm)  Z=0.07500 m (75.00 mm)  [method: BBoxCache]
- **BBox highlight flags**: near_75mm
- **Non-unity ancestor scales**: none detected
- **Highlights**:
  - bbox_Z = 75.00 mm ≈ 75.0 mm (matches CAD board thickness)

### [!] `/World/Board_Tese/Body1`

- **Type**: `Mesh`
- **Local scale**: absent
- **BBox available**: `True`
- **BBox method**: `BBoxCache`
- **BBox (world)**: X=0.20000 m (200.00 mm)  Y=0.17500 m (175.00 mm)  Z=0.07500 m (75.00 mm)  [method: BBoxCache]
- **BBox highlight flags**: near_75mm
- **Non-unity ancestor scales**: none detected
- **Highlights**:
  - bbox_Z = 75.00 mm ≈ 75.0 mm (matches CAD board thickness)

### `/World/Board_Tese/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C`

- **Type**: `Material`
- **Local scale**: absent
- **BBox available**: `False`
- **BBox method**: `empty`
- **BBox (world)**: unavailable [empty] — BBoxCache returned empty range; no mesh points found in subtree
- **Non-unity ancestor scales**: none detected
- **Errors during inspection**:
  - `bbox: BBoxCache returned empty range; no mesh points found in subtree`

### `/World/Board_Tese/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C_shader`

- **Type**: `Shader`
- **Local scale**: absent
- **BBox available**: `False`
- **BBox method**: `empty`
- **BBox (world)**: unavailable [empty] — BBoxCache returned empty range; no mesh points found in subtree
- **Non-unity ancestor scales**: none detected
- **Errors during inspection**:
  - `bbox: BBoxCache returned empty range; no mesh points found in subtree`

### [!] `/World/Circle`

- **Type**: `Xform`
- **Local scale**: (1.000000, 1.000000, 1.000000)  [unity]
- **BBox available**: `True`
- **BBox method**: `BBoxCache`
- **BBox (world)**: X=0.04994 m (49.94 mm)  Y=0.04997 m (49.97 mm)  Z=0.10500 m (105.00 mm)  [method: BBoxCache]
- **BBox highlight flags**: near_51mm
- **Non-unity ancestor scales**: none detected
- **Highlights**:
  - name/path contains 'circle' or 'cylinder'
  - bbox_X = 49.94 mm ≈ 51.0 mm (matches CAD nominal)
  - bbox_Y = 49.97 mm ≈ 51.0 mm (matches CAD nominal)

### [!] `/World/Circle/Body1`

- **Type**: `Mesh`
- **Local scale**: absent
- **BBox available**: `True`
- **BBox method**: `BBoxCache`
- **BBox (world)**: X=0.04994 m (49.94 mm)  Y=0.04997 m (49.97 mm)  Z=0.10500 m (105.00 mm)  [method: BBoxCache]
- **BBox highlight flags**: near_51mm
- **Non-unity ancestor scales**: none detected
- **Highlights**:
  - name/path contains 'circle' or 'cylinder'
  - bbox_X = 49.94 mm ≈ 51.0 mm (matches CAD nominal)
  - bbox_Y = 49.97 mm ≈ 51.0 mm (matches CAD nominal)

### [!] `/World/Circle/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C`

- **Type**: `Material`
- **Local scale**: absent
- **BBox available**: `False`
- **BBox method**: `empty`
- **BBox (world)**: unavailable [empty] — BBoxCache returned empty range; no mesh points found in subtree
- **Non-unity ancestor scales**: none detected
- **Highlights**:
  - name/path contains 'circle' or 'cylinder'
- **Errors during inspection**:
  - `bbox: BBoxCache returned empty range; no mesh points found in subtree`

### [!] `/World/Circle/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C_shader`

- **Type**: `Shader`
- **Local scale**: absent
- **BBox available**: `False`
- **BBox method**: `empty`
- **BBox (world)**: unavailable [empty] — BBoxCache returned empty range; no mesh points found in subtree
- **Non-unity ancestor scales**: none detected
- **Highlights**:
  - name/path contains 'circle' or 'cylinder'
- **Errors during inspection**:
  - `bbox: BBoxCache returned empty range; no mesh points found in subtree`

---

## Forced Inspection Targets

The following four prims are always reported, regardless of search terms.

### `/World/Circle`
- **Found**: yes
- **Type**: `Xform`
- **BBox available**: `True`
- **BBox method**: `BBoxCache`
- **BBox (world)**: X=0.04994 m (49.94 mm)  Y=0.04997 m (49.97 mm)  Z=0.10500 m (105.00 mm)  [method: BBoxCache]
- **BBox highlight flags**: near_51mm

### `/World/Circle/Body1`
- **Found**: yes
- **Type**: `Mesh`
- **BBox available**: `True`
- **BBox method**: `BBoxCache`
- **BBox (world)**: X=0.04994 m (49.94 mm)  Y=0.04997 m (49.97 mm)  Z=0.10500 m (105.00 mm)  [method: BBoxCache]
- **BBox highlight flags**: near_51mm

### `/World/Board_Tese`
- **Found**: yes
- **Type**: `Xform`
- **BBox available**: `True`
- **BBox method**: `BBoxCache`
- **BBox (world)**: X=0.20000 m (200.00 mm)  Y=0.17500 m (175.00 mm)  Z=0.07500 m (75.00 mm)  [method: BBoxCache]
- **BBox highlight flags**: near_75mm

### `/World/Board_Tese/Body1`
- **Found**: yes
- **Type**: `Mesh`
- **BBox available**: `True`
- **BBox method**: `BBoxCache`
- **BBox (world)**: X=0.20000 m (200.00 mm)  Y=0.17500 m (175.00 mm)  Z=0.07500 m (75.00 mm)  [method: BBoxCache]
- **BBox highlight flags**: near_75mm

---

## Candidate Circular Cavity Prims (4 found)

- `/World/Circle` (Xform)
  - Local scale: (1.000000, 1.000000, 1.000000)  [unity]
  - BBox: X=0.04994 m (49.94 mm)  Y=0.04997 m (49.97 mm)  Z=0.10500 m (105.00 mm)  [method: BBoxCache]
  - **[!]** name/path contains 'circle' or 'cylinder'
  - **[!]** bbox_X = 49.94 mm ≈ 51.0 mm (matches CAD nominal)
  - **[!]** bbox_Y = 49.97 mm ≈ 51.0 mm (matches CAD nominal)
- `/World/Circle/Body1` (Mesh)
  - Local scale: absent
  - BBox: X=0.04994 m (49.94 mm)  Y=0.04997 m (49.97 mm)  Z=0.10500 m (105.00 mm)  [method: BBoxCache]
  - **[!]** name/path contains 'circle' or 'cylinder'
  - **[!]** bbox_X = 49.94 mm ≈ 51.0 mm (matches CAD nominal)
  - **[!]** bbox_Y = 49.97 mm ≈ 51.0 mm (matches CAD nominal)
- `/World/Circle/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C` (Material)
  - Local scale: absent
  - BBox: unavailable [empty] — BBoxCache returned empty range; no mesh points found in subtree
  - **[!]** name/path contains 'circle' or 'cylinder'
- `/World/Circle/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C_shader` (Shader)
  - Local scale: absent
  - BBox: unavailable [empty] — BBoxCache returned empty range; no mesh points found in subtree
  - **[!]** name/path contains 'circle' or 'cylinder'

---

## Candidate Board Prims (4 found)

- `/World/Board_Tese` (Xform)
  - Local scale: (1.000000, 1.000000, 1.000000)  [unity]
  - BBox: X=0.20000 m (200.00 mm)  Y=0.17500 m (175.00 mm)  Z=0.07500 m (75.00 mm)  [method: BBoxCache]
  - **[!]** bbox_Z = 75.00 mm ≈ 75.0 mm (matches CAD board thickness)
- `/World/Board_Tese/Body1` (Mesh)
  - Local scale: absent
  - BBox: X=0.20000 m (200.00 mm)  Y=0.17500 m (175.00 mm)  Z=0.07500 m (75.00 mm)  [method: BBoxCache]
  - **[!]** bbox_Z = 75.00 mm ≈ 75.0 mm (matches CAD board thickness)
- `/World/Board_Tese/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C` (Material)
  - Local scale: absent
  - BBox: unavailable [empty] — BBoxCache returned empty range; no mesh points found in subtree
- `/World/Board_Tese/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C_shader` (Shader)
  - Local scale: absent
  - BBox: unavailable [empty] — BBoxCache returned empty range; no mesh points found in subtree

---

## Non-Unity Scales Found (0 prims)

_No non-unity scales detected in any matching prim or its ancestors._

---

## Measured Bounding Boxes Summary

| Prim Path | Type | BBox X (mm) | BBox Y (mm) | BBox Z (mm) | Method | Flags |
|---|---|---|---|---|---|---|
| `/World/Board_Tese` | Xform | 200.00 | 175.00 | 75.00 | BBoxCache | bbox_Z = 75.00 mm ≈ 75.0 mm (matches CAD board thickness) |
| `/World/Board_Tese/Body1` | Mesh | 200.00 | 175.00 | 75.00 | BBoxCache | bbox_Z = 75.00 mm ≈ 75.0 mm (matches CAD board thickness) |
| `/World/Board_Tese/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C` | Material | N/A | N/A | N/A | empty |  |
| `/World/Board_Tese/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C_shader` | Shader | N/A | N/A | N/A | empty |  |
| `/World/Circle` | Xform | 49.94 | 49.97 | 105.00 | BBoxCache | name/path contains 'circle' or 'cylinder'; bbox_X = 49.94 mm ≈ 51.0 mm (match... |
| `/World/Circle/Body1` | Mesh | 49.94 | 49.97 | 105.00 | BBoxCache | name/path contains 'circle' or 'cylinder'; bbox_X = 49.94 mm ≈ 51.0 mm (match... |
| `/World/Circle/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C` | Material | N/A | N/A | N/A | empty | name/path contains 'circle' or 'cylinder' |
| `/World/Circle/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C/XID_SteelSatin_873C466D5CB8B69EEF5D4FD81DF6749C_shader` | Shader | N/A | N/A | N/A | empty | name/path contains 'circle' or 'cylinder' |

---

## Likely Explanation


**Prims with bounding box ≈ 51.0 mm (matches CAD nominal):**
- `/World/Circle` bbox_mm=[49.939, 49.97, 105.0]
- `/World/Circle/Body1` bbox_mm=[49.939, 49.97, 105.0]

---

## Conclusion Candidates

**CONCLUSION**: Circle mesh authored at correct ~51 mm — inflation source must be elsewhere.

---

> **NOTE**: Baseline 1 (geometric matching) remains BLOCKED until the circular cavity is corrected in the scene and recaptured.
