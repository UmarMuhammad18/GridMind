# Sample Run — GridMind

This document shows an example episode from a GridMind agent run.

The agent's objective is:

> Find the key, unlock the door, and reach the goal tile.

---

## Step 0

### Environment State

```text
########
#A..K..#
#..##..#
#..D.G.#
########
```

### Agent Response

**Action**

```json
{
  "action": "MOVE_RIGHT",
  "explanation": "Moving right brings me closer to the key."
}
```

---

## Step 3

### Environment Update

The agent reaches the key tile.

### Agent Response

**Action**

```json
{
  "action": "PICK_UP",
  "explanation": "I am standing on the key, so I should pick it up."
}
```

### Inventory

```json
{
  "has_key": true
}
```

---

## Step 9

### Environment Update

The agent is standing adjacent to the locked door.

### Agent Response

**Action**

```json
{
  "action": "OPEN_DOOR",
  "explanation": "I have the key and I am next to the door."
}
```

### Door State

```json
{
  "door_open": true
}
```

---

## Step 12

### Environment Update

The agent reaches the goal tile.

### Episode Result

```json
{
  "success": true,
  "message": "Agent reached the goal tile."
}
```

---

## Summary

| Metric          | Value  |
| --------------- | ------ |
| Total Steps     | 12     |
| Key Collected   | ✅ Yes  |
| Door Opened     | ✅ Yes  |
| Goal Reached    | ✅ Yes  |
| Episode Success | ✅ True |

The agent successfully completed the task by:

1. Navigating toward the key.
2. Picking up the key.
3. Opening the locked door.
4. Reaching the goal tile.

This demonstrates the complete perception → reasoning → action loop used by GridMind.
