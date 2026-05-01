# CLAUDE.md

## Project

This is a master thesis project about learning perception-action relations based on geometry for robotic insertion tasks.

The experimental setup is inspired by children's shape sorting toys. The system must detect a geometric piece, extract its geometric characteristics, infer which cavity it fits into, estimate the insertion pose, and eventually execute the insertion with a robotic arm.

The current implementation uses NVIDIA Isaac Sim 5.1 running inside a container, accessed through the WebRTC Isaac Sim client.

The current scene contains:
- a board with geometric cavities;
- geometric pieces created in Fusion;
- initial pieces: rectangle, square, circle, star;
- RGB-D/depth capture through Isaac Sim Replicator annotators.

## Main objective

Do not build a simple shape classifier.

The goal is to infer geometric compatibility and insertion affordances. The system should reason about whether a piece can fit into a cavity, under which rotation and approximate pose.

Prefer outputs such as:
- compatible cavity;
- compatibility score;
- best insertion rotation;
- estimated insertion pose;
- failure reason.

Avoid outputs that only say:
- "this is a square";
- "this is a circle";
- "rectangle maps to cavity 0".

## Development strategy

Work in phases.

Current phase:
1. Detect a single visible piece from RGB-D/depth.
2. Segment the piece from the support surface.
3. Generate a real-scale point cloud.
4. Generate a 2D footprint.
5. Save debug outputs.
6. Keep the code clean and modular.

Next phases:
1. Detect cavities.
2. Implement deterministic geometric baseline.
3. Add multi-view piece reconstruction.
4. Add learned compatibility scoring.
5. Integrate robot pick-and-place/insertion.

Do not jump ahead unless explicitly asked.

## Isaac Sim constraints

Code must be practical for Isaac Sim 5.1.

Use:
- omni.replicator.core for RGB/depth capture;
- async execution pattern compatible with Isaac Sim Script Editor;
- USD/pxr APIs when needed;
- OpenCV and NumPy for image/depth processing.

Avoid:
- generic Python that cannot run inside Isaac Sim;
- dependencies that are not likely to be available in the container;
- DeepStream;
- ROS unless explicitly requested;
- reinforcement learning unless explicitly requested.

## Coding style

Use Python.

Write modular scripts with:
- config section at the top;
- clear functions;
- robust error handling;
- useful print messages;
- deterministic defaults;
- output paths under /tmp/shape_insertion/.

Every perception script should save debug files, including:
- RGB image;
- depth visualization;
- segmentation mask;
- overlay/debug image;
- .npy output;
- metadata JSON.

## Geometry rules

For insertion, real scale matters.

Do not normalize point clouds in a way that destroys object size. It is acceptable to:
- center points around the object centroid;
- express points in local coordinates;
- optionally store scale metadata.

It is not acceptable to scale all objects to the same unit size unless this is explicitly part of a controlled experiment.

## Baseline philosophy

The first baseline should be deterministic and geometric, not learned.

Preferred baseline:
- project piece point cloud into XY footprint;
- project cavity into XY footprint;
- test rotations;
- compute overlap, outside ratio, IoU or Chamfer distance;
- select best cavity and rotation.

The baseline is used as a comparison against learned methods.

## Dataset rules

Avoid tiny datasets that only encode known labels.

Do not rely on hardcoded mappings such as:
- rectangle -> cavity 0
- square -> cavity 1

When generating labels, prefer geometric validity:
- does this piece fit into this cavity at this pose?
- does it collide?
- is the alignment acceptable?

## Debugging rules

When a script fails, first check:
1. camera pose and orientation;
2. depth range;
3. support surface depth estimation;
4. segmentation threshold;
5. connected components;
6. coordinate transforms;
7. whether scale was preserved.

Always ask for or inspect output images before making large changes.

## Communication style

Be direct and critical. Do not agree blindly.

When proposing changes, explain:
- what problem it solves;
- what risk it introduces;
- how to validate it;
- what output files should be inspected.

Prefer incremental changes over large rewrites.
