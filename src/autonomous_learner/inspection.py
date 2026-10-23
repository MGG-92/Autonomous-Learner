"""Researcher-only export, never a source of agent observations."""
import json
from pathlib import Path

from .memory import Memory


def export_dot(database: Path, output: Path):
    rows = Memory.inspect_file(database)
    lines = ["digraph evidence {", "  rankdir=LR;"]
    for row in rows:
        label = f"#{row['id']} {row['relation']} ({row['source']})"
        style = "solid" if row["positive"] else "dashed"
        color = "black" if row["positive"] else "red"
        lines.append(f"  {json.dumps(row['subject'])} -> {json.dumps(row['object'])} "
                     f"[label={json.dumps(label)}, style={style}, color={color}];")
    lines.append("}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
