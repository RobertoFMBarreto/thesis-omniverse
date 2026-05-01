# .claude/rules/perception.md
Use this rule when editing perception, point cloud, depth, segmentation or geometry scripts.

The perception pipeline should preserve real-world scale.

For piece detection:
- estimate support surface depth automatically;
- segment objects above the support surface;
- use connected components;
- select the most plausible component;
- export mask, footprint, point cloud and metadata.

For cavity detection:
- cavities are negative geometry, not objects above the surface;
- detect them as regions below the board surface;
- avoid fixed pixel coordinates unless explicitly debugging.

Always produce visual debug images.
