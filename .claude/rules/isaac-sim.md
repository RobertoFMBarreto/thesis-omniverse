# .claude/rules/isaac-sim.md

Use this rule when editing Isaac Sim scripts.

Scripts are expected to run inside Isaac Sim 5.1, often through the Script Editor, using async execution.

Prefer this pattern:

```python
import asyncio

async def main():
    ...

asyncio.ensure_future(main())
```

Use Replicator annotators for sensor capture:

rgb
distance_to_image_plane

Always save debug outputs to /tmp/shape_insertion/.

Do not assume normal desktop Python execution unless the user explicitly says the script runs outside Isaac Sim.

