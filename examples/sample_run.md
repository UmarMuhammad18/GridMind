# Sample Run — GridMind

This is an illustrative episode on the default `key_door` level, showing the
key moments of a successful run. The shortest solution is 11 steps.

The agent's objective is:

> Find the key, then go to the door, open it, and reach the goal tile behind it.

---

## Step 0

### Environment State

```text
########
#A..K..#
#......#
###D####
#....G.#
########
```

The goal `G` is sealed off: the only way in is through the locked door `D`.

### Agent Response

```json
{
  "action": "MOVE_RIGHT",
  "explanation": "I am at [1, 1] and the key is at [4, 1], so I move right toward it.",
  "note": "Key at [4, 1], door at [3, 3], goal at [5, 4]."
}
```

---

## Step 3

### Environment Update

The agent is at `[4, 1]` and the observation reports `"standing_on": "K"`.

### Agent Response

```json
{
  "action": "PICK_UP",
  "explanation": "I am standing on the key, so I pick it up.",
  "note": ""
}
```

### Inventory

```json
{
  "has_key": true
}
```

---

## Step 6

### Environment State

```text
########
#......#
#..A...#
###D####
#....G.#
########
```

### Agent Response

```json
{
  "action": "OPEN_DOOR",
  "explanation": "I have the key and the door is directly below me.",
  "note": ""
}
```

### Environment Update

```text
Door opened.
```

---

## Step 10

### Agent Response

```json
{
  "action": "MOVE_RIGHT",
  "explanation": "The goal is one tile to my right.",
  "note": ""
}
```

### Episode Result

```json
{
  "success": true,
  "message": "Agent reached the goal tile."
}
```

---

## Summary

| Metric          | Value   |
| --------------- | ------- |
| Total Steps     | 11      |
| Optimal Steps   | 11      |
| Key Collected   | ✅ Yes  |
| Door Opened     | ✅ Yes  |
| Goal Reached    | ✅ Yes  |
| Episode Success | ✅ True |

The agent successfully completed the task by:

1. Navigating toward the key.
2. Picking up the key.
3. Opening the locked door.
4. Reaching the goal tile.

To watch a real run step by step, replay its log:

```bash
python replay.py logs/run_<timestamp>_<agent>_<level>.json
```
