#!/usr/bin/env python3
"""
ADHD Thinking Comparison
========================
Compare classic linear reasoning vs concurrent "ADHD-style" divergent reasoning
on the same set of problems.

Designed for local models running an OpenAI-compatible endpoint
(vLLM, llama.cpp server, Ollama with OpenAI compatibility, etc.)

Usage:
    python compare.py
    python compare.py --problems problems/my_problems.json
    python compare.py --model deepseek-r1-distill --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "your-model-name")
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_MODEL)

# Cognitive frames used for the divergent (ADHD) branches
FRAMES = [
    "You are currently hyper-focused on the core problem. Stay tight and rigorous.",
    "You just got distracted by a related but slightly off-topic technical detail. Explore it for a moment, then force yourself back.",
    "A wild tangent just appeared in your mind (possibly from a completely different domain or analogy). Follow it for 2-3 sentences, then snap back.",
    "You suddenly doubt one of your earlier assumptions. Circle back and attack it from a new angle before continuing.",
    "You are thinking like someone with ADHD who keeps bouncing between ideas but eventually weaves them into a coherent conclusion. Show the jumps.",
]

app = typer.Typer(help="Compare linear vs ADHD-style concurrent reasoning")
console = Console()


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

async def normal_thinking(client: AsyncOpenAI, model: str, problem: str) -> str:
    """Classic linear chain-of-thought."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Think step by step in a clean, linear fashion. "
                    "Go straight from the problem to a solid solution. "
                    "Be rigorous and structured."
                ),
            },
            {"role": "user", "content": problem},
        ],
        temperature=0.55,
        max_tokens=1600,
    )
    return resp.choices[0].message.content or ""


async def adhd_branch(
    client: AsyncOpenAI,
    model: str,
    problem: str,
    frame: str,
    branch_id: int,
) -> str:
    """One isolated divergent thought stream."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are simulating an ADHD-style reasoning process.\n"
                    f"{frame}\n\n"
                    "Rules:\n"
                    "- You are allowed (and encouraged) to jump, get distracted, "
                    "circle back, and change angle.\n"
                    "- Explicitly show the jumps in your thinking "
                    '(e.g. "wait, that reminds me of...", "okay coming back...", '
                    '"side thought:", "sudden doubt:").\n'
                    "- Stay coherent enough that a final answer can still be extracted.\n"
                    "- End with a short provisional conclusion from this particular mental path."
                ),
            },
            {"role": "user", "content": problem},
        ],
        temperature=0.85,
        max_tokens=1000,
    )
    content = resp.choices[0].message.content or ""
    return f"[Branch {branch_id}]\n{content}"


async def synthesize(
    client: AsyncOpenAI,
    model: str,
    problem: str,
    branches: List[str],
) -> str:
    """Final synthesizer that weaves the scattered branches into one answer."""
    joined = "\n\n---\n\n".join(branches)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You received several parallel, somewhat scattered thought streams "
                    "from an ADHD-style process.\n\n"
                    "Produce one strong, coherent final answer that:\n"
                    "- Benefits from the jumps and different angles\n"
                    "- Explicitly mentions useful tangents or doubts that improved the answer\n"
                    "- Feels like it came from a mind that couldn't sit still, but still lands cleanly\n"
                    "- Is practical and actionable"
                ),
            },
            {
                "role": "user",
                "content": f"Original problem:\n{problem}\n\nScattered branches:\n{joined}",
            },
        ],
        temperature=0.45,
        max_tokens=1800,
    )
    return resp.choices[0].message.content or ""


async def judge(
    client: AsyncOpenAI,
    judge_model: str,
    problem: str,
    answer_a: str,
    answer_b: str,
) -> Dict[str, Any]:
    """Blind LLM-as-judge. Returns structured scores."""
    prompt = f"""You are a skeptical senior engineer evaluating two solutions to the same problem.

Problem:
{problem}

----- Solution A -----
{answer_a}

----- Solution B -----
{answer_b}

Score each solution from 0 to 10 on these dimensions:
- novelty
- breadth
- trap_detection
- actionability
- coherence
- usefulness

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "A": {{
    "novelty": 0,
    "breadth": 0,
    "trap_detection": 0,
    "actionability": 0,
    "coherence": 0,
    "usefulness": 0
  }},
  "B": {{
    "novelty": 0,
    "breadth": 0,
    "trap_detection": 0,
    "actionability": 0,
    "coherence": 0,
    "usefulness": 0
  }},
  "winner": "A" or "B" or "tie",
  "reason": "one short paragraph explaining the decision"
}}"""

    resp = await client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
        max_tokens=900,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "A": {},
            "B": {},
            "winner": "tie",
            "reason": "Judge returned invalid JSON",
            "raw": raw,
        }


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

async def run_one_comparison(
    client: AsyncOpenAI,
    model: str,
    judge_model: str,
    problem: str,
    num_frames: int = 5,
) -> Dict[str, Any]:
    """Run Normal + ADHD on one problem and judge the results."""
    frames = FRAMES[:num_frames]

    # Launch concurrent work
    normal_task = asyncio.create_task(normal_thinking(client, model, problem))
    branch_tasks = [
        asyncio.create_task(adhd_branch(client, model, problem, frame, i))
        for i, frame in enumerate(frames)
    ]

    normal_result = await normal_task
    branch_results = await asyncio.gather(*branch_tasks)
    adhd_final = await synthesize(client, model, problem, branch_results)

    # Judge (Normal = A, ADHD = B)
    scores = await judge(client, judge_model, problem, normal_result, adhd_final)

    return {
        "problem": problem,
        "normal": normal_result,
        "adhd_branches": branch_results,
        "adhd_final": adhd_final,
        "scores": scores,
        "timestamp": datetime.now().isoformat(),
    }


async def full_comparison(
    client: AsyncOpenAI,
    model: str,
    judge_model: str,
    problems: List[str],
    num_frames: int = 5,
    output_dir: Path = Path("results"),
) -> List[Dict[str, Any]]:
    """Run the full comparison suite and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running comparisons...", total=len(problems))

        for i, problem in enumerate(problems, 1):
            progress.update(
                task,
                description=f"[{i}/{len(problems)}] {problem[:70]}...",
            )
            result = await run_one_comparison(
                client, model, judge_model, problem, num_frames=num_frames
            )
            results.append(result)

            # Live feedback
            winner = result["scores"].get("winner", "?")
            reason = result["scores"].get("reason", "")[:100]
            console.print(f"  → Winner: [bold]{winner}[/bold]  |  {reason}...")

            progress.advance(task)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"comparison_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also write a human-readable summary
    summary_file = output_dir / f"summary_{timestamp}.txt"
    write_summary(results, summary_file)

    console.print(f"\n[green]Detailed results saved to:[/green] {out_file}")
    console.print(f"[green]Human summary saved to:[/green]   {summary_file}")

    return results


def write_summary(results: List[Dict[str, Any]], path: Path) -> None:
    """Write a readable text summary."""
    lines = []
    lines.append("=" * 80)
    lines.append("ADHD Thinking Comparison Summary")
    lines.append("=" * 80)
    lines.append("")

    wins = {"A": 0, "B": 0, "tie": 0}
    for r in results:
        w = r["scores"].get("winner", "tie")
        wins[w] = wins.get(w, 0) + 1

    lines.append(f"Total problems : {len(results)}")
    lines.append(f"Normal (A) wins: {wins['A']}")
    lines.append(f"ADHD   (B) wins: {wins['B']}")
    lines.append(f"Ties           : {wins['tie']}")
    lines.append("")

    for i, r in enumerate(results, 1):
        lines.append("-" * 80)
        lines.append(f"Problem {i}: {r['problem']}")
        lines.append(f"Winner   : {r['scores'].get('winner')}")
        lines.append(f"Reason   : {r['scores'].get('reason', '')}")
        lines.append("")
        lines.append("Scores (A = Normal, B = ADHD):")
        scores = r["scores"]
        for side in ("A", "B"):
            s = scores.get(side, {})
            lines.append(
                f"  {side}: novelty={s.get('novelty')}  breadth={s.get('breadth')}  "
                f"trap={s.get('trap_detection')}  action={s.get('actionability')}  "
                f"coherence={s.get('coherence')}  usefulness={s.get('usefulness')}"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_aggregate(results: List[Dict[str, Any]]) -> None:
    """Pretty print aggregate stats."""
    table = Table(title="Aggregate Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Normal (A)", justify="right")
    table.add_column("ADHD (B)", justify="right")

    wins = {"A": 0, "B": 0, "tie": 0}
    for r in results:
        w = r["scores"].get("winner", "tie")
        wins[w] = wins.get(w, 0) + 1

    table.add_row("Wins", str(wins["A"]), str(wins["B"]))
    table.add_row("Ties", str(wins["tie"]), str(wins["tie"]))

    console.print(table)

    if wins["B"] > wins["A"]:
        console.print(
            Panel(
                "[bold green]ADHD concurrent style won overall[/bold green]\n"
                "The divergent + synthesizer approach produced stronger results "
                "on this problem set.",
                title="Conclusion",
            )
        )
    elif wins["A"] > wins["B"]:
        console.print(
            Panel(
                "[bold yellow]Linear style won overall[/bold yellow]\n"
                "On this particular set the classic approach was preferred.",
                title="Conclusion",
            )
        )
    else:
        console.print(Panel("Results are mixed / tied.", title="Conclusion"))


# ---------------------------------------------------------------------------
# Default problems
# ---------------------------------------------------------------------------

DEFAULT_PROBLEMS = [
    "Design a rate-limiting system that still works well under sudden 100x traffic spikes and partial network partitions.",
    "How would you design a local multi-agent system that can safely self-improve its own prompts over time without drifting into nonsense?",
    "Propose a memory architecture for a 32B model running on 128GB unified memory that needs to keep very long agent sessions (weeks) without catastrophic forgetting of important decisions.",
    "What is a good way to detect and recover when an LLM agent starts hallucinating tool results in a multi-step workflow?",
    "Design a simple but robust way to do online preference learning from user thumbs-up / thumbs-down feedback on a local model without training a large separate reward model.",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    base_url: str = typer.Option(DEFAULT_BASE_URL, help="OpenAI-compatible base URL"),
    api_key: str = typer.Option(DEFAULT_API_KEY, help="API key (can be dummy for local)"),
    model: str = typer.Option(DEFAULT_MODEL, help="Model name for generation"),
    judge_model: str = typer.Option(DEFAULT_JUDGE_MODEL, help="Model used as judge"),
    problems_file: Optional[Path] = typer.Option(
        None, help="JSON file with list of problem strings"
    ),
    num_frames: int = typer.Option(5, help="Number of concurrent ADHD frames"),
    output_dir: Path = typer.Option(Path("results"), help="Directory for results"),
):
    """Run Normal vs ADHD concurrent reasoning comparison."""
    console.print(
        Panel.fit(
            "[bold]ADHD Thinking Comparison[/bold]\n"
            "Linear CoT  vs  Concurrent divergent frames + synthesizer",
            border_style="blue",
        )
    )

    # Load problems
    if problems_file and problems_file.exists():
        with open(problems_file, encoding="utf-8") as f:
            problems = json.load(f)
        if not isinstance(problems, list):
            console.print("[red]problems file must contain a JSON list of strings[/red]")
            raise typer.Exit(1)
    else:
        problems = DEFAULT_PROBLEMS
        console.print("[dim]Using built-in default problems[/dim]")

    console.print(f"Model        : {model}")
    console.print(f"Judge model  : {judge_model}")
    console.print(f"Base URL     : {base_url}")
    console.print(f"Problems     : {len(problems)}")
    console.print(f"ADHD frames  : {num_frames}")
    console.print()

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    results = asyncio.run(
        full_comparison(
            client=client,
            model=model,
            judge_model=judge_model,
            problems=problems,
            num_frames=num_frames,
            output_dir=output_dir,
        )
    )

    print_aggregate(results)


if __name__ == "__main__":
    app()
