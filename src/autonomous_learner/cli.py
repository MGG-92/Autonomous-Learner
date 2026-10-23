import argparse
import csv
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from .agent import Agent, run_episode
from .evaluation import compare, evaluate
from .memory import Memory
from .policy import Heuristic
from .training import TrainConfig, load_checkpoint, save_checkpoint, train
from .world import SyntheticTeacher, World


def demo(path: Path, seed: int, checkpoint: Path | None = None):
    if path.exists():
        raise FileExistsError(f"Demo requires a fresh database; choose another --memory path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    world = World.generate(seed)
    goal = world.goal()
    memory = Memory(path)
    print(f"Goal: {goal.text()}")
    print("Before teaching:", memory.answer(goal).status)
    policy = load_checkpoint(checkpoint)[0] if checkpoint else Heuristic()
    agent = Agent(memory, world.catalog(), goal)
    result, _ = run_episode(agent, SyntheticTeacher(world), policy, world.truth(goal),
                            observe=lambda event: print(json.dumps(event)))
    memory.close()
    memory = Memory(path)
    print("After restart:", json.dumps(memory.answer(goal).to_dict()))
    # A new query about the ancestor's own capability needs a shorter proof.
    leaf = next(f.object for f in world.facts if f.subject == goal.subject and f.relation == "is_a")
    from .types import Query
    transfer = Query(leaf, "can", goal.object)
    print("New query:", transfer.text(), json.dumps(memory.answer(transfer).to_dict()))
    empty = Memory(capacity=0)
    print("Memory ablation:", empty.answer(goal).status)
    empty.close()
    memory.close()
    if not result.correct:
        raise RuntimeError("Demo failed to acquire enough evidence")


def experiment(output: Path, config: TrainConfig):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError("Use a fresh output directory to preserve previous experiments")
    policy, initial, curve = train(config, progress=lambda row: print(json.dumps(row), flush=True))
    save_checkpoint(output / "controller.pt", policy, config)
    save_checkpoint(output / "initial.pt", initial, config)
    with (output / "curve.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    report = {"config": asdict(config), "environment": {"python": platform.python_version(),
              "torch": torch.__version__, "platform": platform.platform(), "git_commit": revision,
              "git_dirty": dirty}, "checkpoint_selection": "maximum validation mean reward; ties keep earlier checkpoint",
              "results": compare(policy, initial, config), "curve": curve,
              "limitations": ["Scripted monotonic reasoning", "Fixed FIFO memory admission/eviction",
                               "Template/structured output; no language model", "Single training seed",
                               "Easy generated task family; evidence status is not calibrated confidence"]}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Controlled autonomous learning experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("demo")
    d.add_argument("--memory", type=Path, default=Path("runs/demo.sqlite"))
    d.add_argument("--seed", type=int, default=900_001)
    d.add_argument("--checkpoint", type=Path)
    e = commands.add_parser("experiment")
    e.add_argument("--output", type=Path, default=Path("runs/experiment"))
    e.add_argument("--config", type=Path)
    i = commands.add_parser("inspect")
    i.add_argument("memory", type=Path)
    v = commands.add_parser("evaluate")
    v.add_argument("checkpoint", type=Path)
    v.add_argument("--start", type=int, default=300_000)
    v.add_argument("--worlds", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "demo":
        demo(args.memory, args.seed, args.checkpoint)
    elif args.command == "experiment":
        config = TrainConfig(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else TrainConfig()
        experiment(args.output, config)
    elif args.command == "inspect":
        if not args.memory.is_file():
            parser.error("memory database does not exist")
        print(json.dumps(Memory.inspect_file(args.memory), indent=2))
    elif args.command == "evaluate":
        policy, config = load_checkpoint(args.checkpoint)
        print(json.dumps(evaluate(policy, args.start, args.worlds, config.max_steps), indent=2))
