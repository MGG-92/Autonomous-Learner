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
    selected = max(curve, key=lambda row: row["mean_reward"])
    report["selected_checkpoint"] = {"stage": selected["stage"], "iteration": selected["iteration"]}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], indent=2))
    return report


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
    p = commands.add_parser("replicate", help="Repeat a frozen experiment over optimizer seeds")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 97])
    p.add_argument("--output", type=Path, default=Path("runs/replication"))
    m = commands.add_parser("memory-experiment")
    m.add_argument("--output", type=Path, default=Path("runs/memory-experiment"))
    dm = commands.add_parser("delayed-memory", help="Learn memory control without the specific future goal")
    dm.add_argument("--config", type=Path, default=Path("configs/delayed-memory.json"))
    dm.add_argument("--output", type=Path, default=Path("runs/delayed-memory"))
    dm.add_argument("--phase", choices=["validation", "final"], default="validation")
    rules = commands.add_parser("rule-discovery", help="Infer unfamiliar finite rules without a language model")
    rules.add_argument("--config", type=Path, default=Path("configs/rule-discovery.json"))
    rules.add_argument("--output", type=Path, required=True)
    rules.add_argument("--phase", choices=["validation", "final"], default="validation")
    qp = commands.add_parser("learned-investigation", help="Train a development-only model-based query policy")
    qp.add_argument("--config", type=Path, default=Path("configs/learned-investigation.json"))
    qp.add_argument("--output", type=Path, required=True)
    ad = commands.add_parser("action-model-demo", help="Scripted transition-belief software demo, not a benchmark")
    ad.add_argument("--output", type=Path, required=True)
    ap = commands.add_parser("action-prediction", help="Development-only action-world prediction comparison")
    ap.add_argument("--config", type=Path, default=Path("configs/action-prediction.json"))
    ap.add_argument("--output", type=Path, required=True)
    ag = commands.add_parser("action-goals", help="Development-only open-loop goal comparison")
    ag.add_argument("--config", type=Path, default=Path("configs/action-goals.json"))
    ag.add_argument("--output", type=Path, required=True)
    dr = commands.add_parser("delayed-robustness", help="Evaluate frozen memory policies under resource/stream shifts")
    dr.add_argument("--checkpoints", type=Path, required=True)
    dr.add_argument("--output", type=Path, required=True)
    dr.add_argument("--start", type=int, default=2_100_000)
    dr.add_argument("--worlds", type=int, default=100)
    r = commands.add_parser("run", help="Run bounded autonomous ticks; repeat to resume")
    r.add_argument("--session", type=Path, default=Path("runs/session.json"))
    r.add_argument("--seed", type=int, default=900_001)
    r.add_argument("--checkpoint", type=Path)
    r.add_argument("--retention-checkpoint", type=Path)
    r.add_argument("--teacher", choices=["synthetic", "human", "local"], default="synthetic")
    r.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    r.add_argument("--ticks", type=int, default=8)
    r.add_argument("--interval", type=float, default=0.0)
    r.add_argument("--max-steps", type=int, default=8)
    r.add_argument("--max-questions", type=int, default=8)
    i = commands.add_parser("inspect")
    i.add_argument("memory", type=Path)
    g = commands.add_parser("graph", help="Export inspectable evidence as Graphviz DOT")
    g.add_argument("memory", type=Path)
    g.add_argument("--output", type=Path, default=Path("runs/memory.dot"))
    v = commands.add_parser("evaluate")
    v.add_argument("checkpoint", type=Path)
    v.add_argument("--start", type=int, default=300_000)
    v.add_argument("--worlds", type=int, default=100)
    l = commands.add_parser("language-smoke", help="Download/load optional local model and inspect candidates")
    l.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    l.add_argument("--revision")
    l.add_argument("--local-only", action="store_true")
    l.add_argument("--output", type=Path, default=Path("runs/language-smoke.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "demo":
        demo(args.memory, args.seed, args.checkpoint)
    elif args.command == "experiment":
        config = TrainConfig(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else TrainConfig()
        experiment(args.output, config)
    elif args.command == "replicate":
        from dataclasses import replace
        if len(set(args.seeds)) != len(args.seeds) or args.output.exists():
            parser.error("Use unique optimizer seeds and a fresh output directory")
        config = TrainConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        reports = [experiment(args.output / f"seed-{seed}", replace(config, seed=seed)) for seed in args.seeds]
        summary = {"optimizer_seeds": args.seeds, "config": asdict(config),
                   "results_by_seed": {str(seed): report["results"] for seed, report in zip(args.seeds, reports)},
                   "limitations": ["Shared generated test worlds across optimizer seeds",
                                    "Three seeds are preliminary; do not infer broad robustness",
                                    "Task family and supervised proof demonstrations remain simple"]}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    elif args.command == "memory-experiment":
        from .memory_policy import memory_experiment
        print(json.dumps(memory_experiment(args.output), indent=2))
    elif args.command == "delayed-memory":
        from .delayed_training import DelayedConfig, experiment as delayed_experiment
        config = DelayedConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        report = delayed_experiment(config, args.output, args.phase)
        print(json.dumps({"gate": report["gate"], "results": report["results_by_seed"]}, indent=2))
    elif args.command == "rule-discovery":
        from .rule_experiment import RuleConfig, run as rule_experiment
        config = RuleConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        report = rule_experiment(config, args.output, args.phase)
        print(json.dumps(report["summary"], indent=2))
    elif args.command == "learned-investigation":
        from .query_training import QueryConfig, run as query_experiment
        config = QueryConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        report = query_experiment(config, args.output)
        print(json.dumps(report["results_by_seed"], indent=2))
    elif args.command == "action-model-demo":
        from .action_demo import run as action_demo
        report = action_demo(args.output)
        print(json.dumps({"kind": report["kind"], "real_actions": report["real_actions"],
                          "candidate_counts": report["candidate_counts_after_experience"],
                          "before": report["before"]["probabilities"],
                          "after_experience": report["after_experience"]["probabilities"],
                          "after_reset": report["after_reset"]["probabilities"]}, indent=2))
    elif args.command == "action-prediction":
        from .action_prediction import PredictionConfig, run as action_prediction
        config = PredictionConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        report = action_prediction(config, args.output, progress=lambda row: print(json.dumps(row), flush=True))
        print(json.dumps(report["results_by_seed"], indent=2))
    elif args.command == "action-goals":
        from .action_goals import GoalConfig, run as action_goals
        config = GoalConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        report = action_goals(config, args.output, progress=lambda row: print(json.dumps(row), flush=True))
        print(json.dumps(report["results_by_seed"], indent=2))
    elif args.command == "delayed-robustness":
        from .delayed_robustness import robustness
        robustness(args.checkpoints, args.output, args.start, args.worlds)
    elif args.command == "run":
        from .loop import PersistentLoop
        from .teachers import HumanTeacher, LocalLMTeacher
        world = World.generate(args.seed)
        args.session.parent.mkdir(parents=True, exist_ok=True)
        expected_type = "DualMemory" if args.retention_checkpoint else "Memory"
        memory_paths = ([Path(str(args.session.with_suffix("")) + suffix)
                         for suffix in (".episodes.sqlite", ".beliefs.sqlite")]
                        if args.retention_checkpoint else [args.session.with_suffix(".sqlite")])
        if args.session in memory_paths:
            parser.error("Session JSON and evidence database paths must be different")
        if args.session.exists():
            state = json.loads(args.session.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or type(state.get("schema_version")) is not int or state["schema_version"] != 2:
                parser.error("Preserve legacy session files and resume with their original source; use a fresh path here")
            signature = state.get("signature")
            if not isinstance(signature, dict) or not isinstance(signature.get("memory"), dict):
                parser.error("Invalid session signature; preserve files and investigate")
            if signature["memory"].get("type") != expected_type or any(not p.is_file() for p in memory_paths):
                parser.error("Session memory type differs or an evidence database is missing; preserve files and investigate")
        if args.retention_checkpoint:
            from .memory_policy import DualMemory, load_retention
            memory = DualMemory(world.goal(), load_retention(args.retention_checkpoint),
                                path=args.session.with_suffix(""))
        else:
            memory = Memory(args.session.with_suffix(".sqlite"))
        try:
            policy = load_checkpoint(args.checkpoint)[0] if args.checkpoint else Heuristic()
            agent = Agent(memory, world.catalog(), world.goal(), args.max_steps)
            loop = PersistentLoop(args.session, agent, policy, args.max_questions)
            if args.teacher == "synthetic":
                teacher = SyntheticTeacher(world)
            elif args.teacher == "human":
                teacher = HumanTeacher()
            else:
                from .language import FrozenLanguageModel
                teacher = LocalLMTeacher(FrozenLanguageModel(args.model), json.dumps([asdict(f) for f in world.facts]))
            for event in loop.run(teacher, args.ticks, args.interval):
                print(json.dumps(event), flush=True)
        finally:
            memory.close()
    elif args.command == "inspect":
        if not args.memory.is_file():
            parser.error("memory database does not exist")
        print(json.dumps(Memory.inspect_file(args.memory), indent=2))
    elif args.command == "graph":
        from .inspection import export_dot
        if not args.memory.is_file():
            parser.error("memory database does not exist")
        if args.output.exists():
            parser.error("Choose a fresh graph output path")
        export_dot(args.memory, args.output)
        print(f"Wrote {args.output}")
    elif args.command == "evaluate":
        policy, config = load_checkpoint(args.checkpoint)
        print(json.dumps(evaluate(policy, args.start, args.worlds, config.max_steps), indent=2))
    elif args.command == "language-smoke":
        from .language import FrozenLanguageModel, render_answer
        from .types import Answer, Query
        from .token_controller import TokenController, controlled_generate
        import time
        if args.output.exists():
            parser.error("Choose a new language-smoke output path")
        start = time.perf_counter()
        engine = FrozenLanguageModel(args.model, local_only=args.local_only, revision=args.revision)
        cached = engine.candidates("The agent learns by", page_size=200, max_pages=2)
        first, second = cached.page(0), cached.page(1)
        assert engine.forward_calls == 1
        assert not set(first.token_ids.tolist()) & set(second.token_ids.tolist())
        controller = TokenController(first.embeddings.shape[1], 4)
        choice = int(controller(first, torch.zeros(4)).argmax())
        report = {"model": args.model, "forward_calls_for_two_pages": engine.forward_calls,
                  "model_revision": getattr(engine.model.config, "_commit_hash", None),
                  "page_sizes": [len(first.token_ids), len(second.token_ids)],
                  "all_parameters_frozen": all(not p.requires_grad for p in engine.model.parameters()),
                  "candidate_examples": [{"id": int(first.token_ids[i]), "text": first.text[i],
                        "probability": float(first.log_probabilities[i].exp())} for i in range(5)],
                  "untrained_controller_choice": first.text[choice] if choice < len(first.text) else ["MORE", "END"][choice - len(first.text)],
                  "render": render_answer(engine, Query("dax", "can", "zep"), Answer("SUPPORTED", (1, 2, 3))),
                  "controlled_generation": controlled_generate(engine, controller, torch.zeros(4),
                                               "The agent learns by", max_new_tokens=8),
                  "elapsed_seconds": time.perf_counter() - start,
                  "limitations": ["Untrained token scorer initialized to the language prior",
                                   "Strict template acceptance; language benefit not established",
                                   "Each new context is recomputed; only candidate pagination is cached"]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
