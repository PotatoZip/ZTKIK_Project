"""Phase 6 – render ``report/report.md`` from a template whose numbers are placeholders resolved
from the run artifacts, so every figure in the report is traceable to a JSON file on disk.

Placeholder syntax (inside ``{{ … }}``)::

    {{ ${runs/ae_base:at_threshold.recall} | pct1 }}
    {{ ${runs/ae_base:at_threshold.recall} - ${runs/baseline_pca_k5:at_threshold.recall} | pp1 }}
    {{ mean(${runs/ae_base_seed0:at_threshold.recall}, ${runs/ae_base:roc_auc}) | pct1 }}

* ``${<path>:<key.path>}`` reads a JSON value. ``<path>`` is relative to the project root; a
  directory means its ``metrics.json``, otherwise the file itself (``runs/ae_base/threshold.json``,
  ``data/processed/data_report.json``). ``<key.path>`` is dot-separated; integers index lists.
* The expression may combine refs and numbers with ``+ - * /`` and ``mean/std/min/max/abs``
  (``std`` is the sample standard deviation).
* ``| fmt`` is one of ``pct0..pct3`` (``77.2 %``), ``pp1`` (percentage points), ``f0..f4``,
  ``int`` (thousands separated by a space), ``verdict`` (``met`` / ``not met``), ``str``.
  Without a format, numbers are printed with ``:g`` and strings verbatim.
"""

from __future__ import annotations

import ast
import json
import re
import statistics
from pathlib import Path
from typing import Any

PLACEHOLDER = re.compile(r"\{\{(.+?)\}\}", re.DOTALL)
REF = re.compile(r"\$\{([^}:]+):([^}]+)\}")
DEFAULT_TEMPLATE = "report/report_template.md"
DEFAULT_OUTPUT = "report/report.md"

_FUNCS = {
    "mean": statistics.fmean,
    "std": statistics.stdev,
    "min": min,
    "max": max,
    "abs": abs,
}


class ReportError(ValueError):
    """Unresolvable placeholder (missing file, key or bad format)."""


class ArtifactStore:
    """Lazy JSON loader rooted at the project directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._cache: dict[Path, Any] = {}

    def file_for(self, path: str) -> Path:
        p = self.root / path
        if p.is_dir():
            p = p / "metrics.json"
        return p

    def load(self, path: str) -> Any:
        p = self.file_for(path)
        if p not in self._cache:
            if not p.exists():
                raise ReportError(f"artifact {p} not found")
            self._cache[p] = json.loads(p.read_text())
        return self._cache[p]

    def get(self, path: str, keys: str) -> Any:
        value = self.load(path)
        for k in keys.strip().split("."):
            try:
                value = value[int(k)] if isinstance(value, list) else value[k]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ReportError(f"key {keys!r} not found in {self.file_for(path)}") from exc
        return value


def format_value(value: Any, fmt: str | None) -> str:
    if fmt is None or fmt == "str":
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)
    if fmt == "verdict":
        return "met" if value else "not met"
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReportError(f"format {fmt!r} needs a number, got {value!r}")
    if m := re.fullmatch(r"pct(\d)", fmt):
        return f"{value * 100:.{m.group(1)}f} %"
    if m := re.fullmatch(r"pp(\d)", fmt):
        return f"{value * 100:.{m.group(1)}f} pp"
    if m := re.fullmatch(r"f(\d)", fmt):
        return f"{value:.{m.group(1)}f}"
    if fmt == "int":
        return f"{int(round(value)):,}".replace(",", " ")
    raise ReportError(f"unknown format {fmt!r}")


def _safe_eval(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand, names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.Div):
        a, b = _safe_eval(node.left, names), _safe_eval(node.right, names)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        return a / b
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        args = [_safe_eval(a, names) for a in node.args]
        return (
            _FUNCS[node.func.id](args)
            if node.func.id in ("mean", "std")
            else _FUNCS[node.func.id](*args)
        )
    raise ReportError(f"unsupported expression element: {ast.dump(node)}")


def evaluate_placeholder(body: str, store: ArtifactStore) -> str:
    """Resolve one ``{{ … }}`` body to its final text."""
    expr, fmt = (body.rsplit("|", 1) + [None])[:2] if "|" in body else (body, None)
    expr, fmt = expr.strip(), fmt.strip() if fmt else None
    names: dict[str, Any] = {}

    def sub(m: re.Match) -> str:
        name = f"_r{len(names)}"
        names[name] = store.get(m.group(1).strip(), m.group(2))
        return name

    python_expr = REF.sub(sub, expr)
    if not names:
        raise ReportError(f"placeholder has no artifact reference: {body.strip()!r}")
    if python_expr.strip() in names:  # single ref – may be a string / bool
        return format_value(names[python_expr.strip()], fmt)
    try:
        value = _safe_eval(ast.parse(python_expr, mode="eval"), names)
    except (SyntaxError, TypeError, statistics.StatisticsError) as exc:
        raise ReportError(f"cannot evaluate {body.strip()!r}: {exc}") from exc
    return format_value(value, fmt)


def render(template: str, root: Path) -> str:
    store = ArtifactStore(root)
    errors: list[str] = []

    def repl(m: re.Match) -> str:
        try:
            return evaluate_placeholder(m.group(1), store)
        except ReportError as exc:
            errors.append(str(exc))
            return m.group(0)

    out = PLACEHOLDER.sub(repl, template)
    if errors:
        raise ReportError("\n".join(errors))
    return out


def references(template: str) -> list[tuple[str, str]]:
    """All ``(path, keys)`` pairs referenced by a template (for static checks)."""
    return [(p.strip(), k.strip()) for p, k in REF.findall(template)]


def render_report(
    root: Path, template: str | Path = DEFAULT_TEMPLATE, out: str | Path = DEFAULT_OUTPUT
) -> Path:
    root = Path(root)
    tpl = root / template
    if not tpl.exists():
        raise FileNotFoundError(f"{tpl} not found")
    text = render(tpl.read_text(), root)
    out_path = root / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return out_path
