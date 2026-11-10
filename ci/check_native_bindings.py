# Copyright Xingyu Chen.
# Generate the native binding manifest from source and check binding ownership and liveness.

"""Generate the native binding manifest from source and check binding ownership and liveness."""

# ``--write`` regenerates ``ci/native-binding-manifest.json`` from the pybind registrations under
# ``native/channel`` and the Python dispatch sites under ``witwin/channel``. The default mode
# regenerates it in memory, prints a diff against the committed file, and checks the invariants
# that are derived from the source on every run:
#
# * every pybind symbol is registered once and has a name without a generation or WIP marker;
# * every symbol is dispatched from exactly one Python module (its owner);
# * every owner is reachable from production code: the public API snapshot, or module-level code,
#   through name references in ``witwin/channel`` (tests do not count as callers).
#
# The manifest records signatures and owners only. It holds no line numbers and no hashes, so
# moving a symbol between files changes nothing, and adding one changes code plus a regenerated file.

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass, field
import difflib
import functools
import json
from pathlib import Path
import re
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Direct script execution starts with only ci/ on sys.path.
from tools.refactor_baseline import binding_manifest  # noqa: E402

MANIFEST_PATH = Path("ci/native-binding-manifest.json")
PUBLIC_SNAPSHOT_PATH = Path("ci/public-api-snapshot.json")
PACKAGE_PATH = Path("witwin/channel")
SCHEMA_VERSION = 2
GENERATOR = "python ci/check_native_bindings.py --write"

# Calls whose string argument names the native symbol they resolve.
DISPATCH_CALLS = frozenset(
    {"required_symbol", "_required_native_op", "optional_symbol", "has_symbol", "_required_symbol"}
)
_FORBIDDEN_NAME = re.compile(r"(?:^|_)(?:v\d+|wip)(?:_|$)", re.IGNORECASE)

# Symbols whose Python facade has no production caller. They are kept until their facades and
# bindings are deleted; the list may only shrink, and an entry that becomes live or disappears
# fails the check so it is removed here in the same change.
KNOWN_UNCALLED_SYMBOLS = frozenset(
    {
        "bdpt_apply_los_visibility",
        "bdpt_component_map_buffer",
        "bdpt_empty_subpath_state",
        "bdpt_face_material_tensors",
        "bdpt_face_material_tensors_from_host",
        "bdpt_los_component_maps",
        "bdpt_los_export",
        "bdpt_los_visibility_inputs",
        "bdpt_mis_weights",
        "bdpt_pack_vec3",
        "bdpt_point_component_power",
        "bdpt_store_component_map",
        "bdpt_store_point_component_column",
        "bdpt_store_scaled_component_map",
        "core_diffraction_edge_count",
        "deterministic_field_from_power_phase",
        "deterministic_gather_topology_block",
        "deterministic_los_field",
        "deterministic_phase_from_length",
        "deterministic_reflect_points",
        "deterministic_repeat_range",
        "deterministic_sort_order",
        "deterministic_surface_face_groups",
        "mc_diffraction_state_pack",
        "mc_los_component_maps",
        "path_diffraction_block",
        "path_filter_block",
        "path_filter_los",
        "path_finalize_blocks",
        "path_merge_blocks",
        "path_reflection_candidates",
        "rayd_intersect_backward",
        "rayd_intersect_jvp",
        "rayd_trace_reflections_backward",
        "rayd_trace_reflections_forward_tape",
        "rayd_trace_reflections_jvp",
        "scattering_chain_realization_eval",
        "scattering_chain_realization_eval_backward",
        "scattering_chain_realization_eval_jvp",
    }
)
# Symbols dispatched from more than one module. Same shrink-only rule as above.
KNOWN_SPLIT_OWNER_SYMBOLS = frozenset({"bdpt_accumulate_connection_samples"})


def _module_name(path: Path, package_root: Path) -> str:
    parts = list(path.relative_to(package_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("witwin", "channel", *parts))


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


@dataclass
class PackageIndex:
    """Name-level index of definitions, references and native dispatch sites."""

    symbols: frozenset[str]
    parent: dict[str, str | None] = field(default_factory=dict)
    loads: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    sites: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def top(self, context: str) -> str:
        """Return the module-level definition that encloses ``context``; methods map to classes."""

        while self.parent.get(context):
            context = self.parent[context]  # type: ignore[assignment]
        return context


class _Indexer(ast.NodeVisitor):
    def __init__(self, index: PackageIndex, module: str, dynamic_dispatch: bool) -> None:
        self.index = index
        self.module = module
        self.dynamic_dispatch = dynamic_dispatch
        self.stack: list[str] = []
        self.aliases: dict[str, str] = {}

    def _context(self) -> str:
        return ".".join((self.module, *self.stack)) if self.stack else f"{self.module}:<module>"

    def _enter(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified = ".".join((self.module, *self.stack, node.name))
        self.index.parent[qualified] = ".".join((self.module, *self.stack)) if self.stack else None
        self.stack.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        for child in (*node.decorator_list, *node.bases, *(item.value for item in node.keywords)):
            self.visit(child)
        self._enter(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:  # noqa: N802
        for child in node.decorator_list:
            self.visit(child)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self._enter(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            return
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        for alias in node.names:
            if alias.asname:
                self.aliases[alias.asname] = alias.name

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = _call_name(node.func)
        if name in DISPATCH_CALLS:
            for argument in node.args:
                if isinstance(argument, ast.Constant) and argument.value in self.index.symbols:
                    self.index.sites[argument.value].add(self._context())
        if (
            name == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            self.index.loads[node.args[1].value].add(self._context())
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        # A module that resolves a symbol name held in a variable owns every symbol literal it holds.
        if self.dynamic_dispatch and isinstance(node.value, str) and node.value in self.index.symbols:
            self.index.sites[node.value].add(self._context())

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        self.index.loads[node.attr].add(self._context())
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        self.index.loads[self.aliases.get(node.id, node.id)].add(self._context())


def _has_dynamic_dispatch(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and _call_name(node.func) in DISPATCH_CALLS
        and bool(node.args)
        and not isinstance(node.args[0], ast.Constant)
        for node in ast.walk(tree)
    )


def index_package(repo: Path, symbols: frozenset[str]) -> PackageIndex:
    index = PackageIndex(symbols=symbols)
    package_root = repo / PACKAGE_PATH
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        _Indexer(index, _module_name(path, package_root), _has_dynamic_dispatch(tree)).visit(tree)
    return index


def _public_roots(repo: Path) -> set[str]:
    snapshot = json.loads((repo / PUBLIC_SNAPSHOT_PATH).read_text(encoding="utf-8"))
    return {str(entry["target"]) for module in snapshot["modules"] for entry in module["exports"]}


def live_definitions(index: PackageIndex, roots: set[str]) -> set[str]:
    """Return module-level definitions reachable from module code or the public API by name."""

    by_short_name: dict[str, list[str]] = defaultdict(list)
    for name, parent in index.parent.items():
        if parent is None:
            by_short_name[name.rsplit(".", 1)[1]].append(name)
    # Edges run from the definition (or module body) that holds a reference to every top-level
    # definition carrying the referenced name.
    edges: dict[str, set[str]] = defaultdict(set)
    for short_name, contexts in index.loads.items():
        targets = by_short_name.get(short_name)
        if not targets:
            continue
        for context in contexts:
            source = index.top(context)
            edges[source].update(target for target in targets if target != source)
    frontier = [root for root in roots if root in index.parent]
    frontier.extend(source for source in edges if source.endswith(":<module>"))
    live: set[str] = set()
    while frontier:
        current = frontier.pop()
        if current in live:
            continue
        live.add(current)
        frontier.extend(edges.get(current, ()))
    return live


def _owners(index: PackageIndex, symbol: str) -> list[str]:
    owners = set()
    for context in index.sites.get(symbol, ()):
        owner = index.top(context)
        owners.add(owner.removesuffix(":<module>"))
    return sorted(owners)


def _owner_module(index: PackageIndex, owner: str) -> str:
    return owner if owner not in index.parent else owner.rsplit(".", 1)[0]


@functools.lru_cache(maxsize=4)
def _parsed(repo: Path) -> tuple[dict[str, Any], PackageIndex]:
    raw = binding_manifest(repo)
    symbols = frozenset(str(entry["name"]) for entry in raw["symbols"])
    return raw, index_package(repo, symbols)


def generate_manifest(repo: Path) -> dict[str, Any]:
    raw, index = _parsed(repo)
    rows = []
    for entry in raw["symbols"]:
        rows.append(
            {
                "name": entry["name"],
                "target": entry["target"],
                "return_type": entry["return_type"],
                "parameters": [
                    [parameter["name"], parameter["type"], parameter["default"]]
                    for parameter in entry["parameters"]
                ],
                "pybind_arguments": [
                    [argument["name"], argument["default"]] for argument in entry["pybind_arguments"]
                ],
                "doc": entry["doc"],
                "python_owners": _owners(index, str(entry["name"])),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATOR,
        "module": raw["module"],
        "symbols": rows,
    }


def render_manifest(manifest: dict[str, Any]) -> str:
    """Render one key per line and one parameter per line so diffs stay reviewable."""

    def dump(value: object) -> str:
        return json.dumps(value, ensure_ascii=False)

    lines = ["{"]
    for key in ("schema_version", "generated_by", "module"):
        lines.append(f"  {dump(key)}: {dump(manifest[key])},")
    lines.append('  "symbols": [')
    for position, row in enumerate(manifest["symbols"]):
        lines.append("    {")
        for key in ("name", "target", "return_type", "doc"):
            lines.append(f"      {dump(key)}: {dump(row[key])},")
        for key in ("parameters", "pybind_arguments"):
            items = row[key]
            if not items:
                lines.append(f"      {dump(key)}: [],")
                continue
            lines.append(f"      {dump(key)}: [")
            lines.extend(
                f"        {dump(item)}{',' if index < len(items) - 1 else ''}"
                for index, item in enumerate(items)
            )
            lines.append("      ],")
        lines.append(f'      "python_owners": {dump(row["python_owners"])}')
        lines.append("    }" + ("," if position < len(manifest["symbols"]) - 1 else ""))
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def check_invariants(repo: Path) -> list[str]:
    raw, index = _parsed(repo)
    problems = [f"pybind symbol registered more than once: {name}" for name in raw["duplicate_symbols"]]
    symbols = frozenset(str(entry["name"]) for entry in raw["symbols"])
    for entry in raw["symbols"]:
        for label in ("name", "target"):
            if _FORBIDDEN_NAME.search(str(entry[label]).rsplit("::", 1)[-1]):
                problems.append(f"binding {label} carries a generation or WIP marker: {entry[label]}")

    live = live_definitions(index, _public_roots(repo))
    uncalled: set[str] = set()
    split: set[str] = set()
    for symbol in sorted(symbols):
        owners = _owners(index, symbol)
        if not owners:
            problems.append(f"native symbol has no Python dispatch site: {symbol}")
            continue
        if len({_owner_module(index, owner) for owner in owners}) > 1:
            split.add(symbol)
        if not any(owner not in index.parent or owner in live for owner in owners):
            uncalled.add(symbol)

    problems.extend(
        f"native symbol has no production caller: {name}"
        for name in sorted(uncalled - KNOWN_UNCALLED_SYMBOLS)
    )
    problems.extend(
        f"KNOWN_UNCALLED_SYMBOLS entry is stale (now called or deleted): {name}"
        for name in sorted(KNOWN_UNCALLED_SYMBOLS - uncalled)
    )
    problems.extend(
        f"native symbol is dispatched from more than one module: {name}"
        for name in sorted(split - KNOWN_SPLIT_OWNER_SYMBOLS)
    )
    problems.extend(
        f"KNOWN_SPLIT_OWNER_SYMBOLS entry is stale (now single-owner or deleted): {name}"
        for name in sorted(KNOWN_SPLIT_OWNER_SYMBOLS - split)
    )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--write", action="store_true", help="regenerate the committed manifest")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    manifest_path = repo / MANIFEST_PATH
    rendered = render_manifest(generate_manifest(repo))

    if args.write:
        manifest_path.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"wrote {MANIFEST_PATH.as_posix()}")
    problems = check_invariants(repo)
    committed = manifest_path.read_text(encoding="utf-8") if manifest_path.is_file() else ""
    if committed != rendered:
        diff = difflib.unified_diff(
            committed.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=f"{MANIFEST_PATH.as_posix()} (committed)",
            tofile=f"{MANIFEST_PATH.as_posix()} (generated)",
            n=2,
        )
        sys.stdout.writelines(diff)
        problems.append(f"{MANIFEST_PATH.as_posix()} is out of date; run: {GENERATOR}")
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"native bindings: {len(json.loads(rendered)['symbols'])} symbols, all invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
