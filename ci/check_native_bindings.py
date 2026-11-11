# Copyright Xingyu Chen.
# Generate the native binding manifest from source and check binding ownership and liveness.

"""Generate the native binding manifest from source and check binding ownership and liveness."""

# ``--write`` regenerates ``ci/native-binding-manifest.json`` from the pybind registrations under
# ``native/channel`` and the Python dispatch sites under ``witwin/channel``. The default mode
# regenerates it in memory, prints a diff against the committed file, and checks the invariants
# that are derived from the source on every run:
#
# * every pybind symbol is registered once and has a name without a generation or work-in-progress marker;
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
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType
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
_FORBIDDEN_NAME = re.compile(r"(?:^|_)(?:v\d+|w[i]p)(?:_|$)", re.IGNORECASE)

# Symbols dispatched from more than one module. The list may only shrink, and an entry that
# becomes single-owner or disappears fails the check so it is removed here in the same change.
KNOWN_SPLIT_OWNER_SYMBOLS = frozenset({"bdpt_accumulate_connection_samples"})
# ``NativeOps.op`` declares one native operation's ABI argument list (witwin/channel/kernels/spec.py).
SPEC_DECLARATION = "op"


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


def _spec_symbol(node: ast.AST, symbols: frozenset[str]) -> str | None:
    """The symbol of a native-op declaration ``<ops>.op("symbol", ...)``, else None."""

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == SPEC_DECLARATION
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in symbols
    ):
        return str(node.args[0].value)
    return None


def spec_declarations(tree: ast.Module, symbols: frozenset[str]) -> dict[str, str]:
    """Module-level ``NAME = <ops>.op("symbol", ...)`` declarations, as name -> symbol."""

    declared: dict[str, str] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            symbol = _spec_symbol(statement.value, symbols)
            if symbol is not None and isinstance(target, ast.Name):
                declared[target.id] = symbol
    return declared


class _Indexer(ast.NodeVisitor):
    def __init__(
        self, index: PackageIndex, module: str, dynamic_dispatch: bool, specs: dict[str, str],
    ) -> None:
        self.index = index
        self.module = module
        self.dynamic_dispatch = dynamic_dispatch
        self.specs = specs
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
        if _spec_symbol(node, self.index.symbols) is not None:
            # A declaration names its symbol and composes other declarations' argument lists;
            # neither dispatches. The definitions that reference the declaration dispatch.
            for argument in node.args[1:]:
                self._visit_declaration_argument(argument)
            return
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
        if isinstance(node.value, ast.Name) and node.value.id in self.specs:
            # Reading a declaration's names or checks does not dispatch it.
            self.index.loads[node.value.id].add(self._context())
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        self.index.loads[self.aliases.get(node.id, node.id)].add(self._context())
        if node.id in self.specs and isinstance(node.ctx, ast.Load):
            self.index.sites[self.specs[node.id]].add(self._context())

    def _visit_declaration_argument(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id not in self.specs:
                self.index.loads[self.aliases.get(child.id, child.id)].add(self._context())


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
        _Indexer(
            index, _module_name(path, package_root), _has_dynamic_dispatch(tree),
            spec_declarations(tree, symbols),
        ).visit(tree)
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


def _load_module(repo: Path, path: Path, module_name: str) -> ModuleType:
    """Import a repository module; a module of another tree is executed from its file."""

    if repo == REPOSITORY_ROOT:
        return importlib.import_module(module_name)
    loader_spec = importlib.util.spec_from_file_location(f"_binding_check.{module_name}", path)
    if loader_spec is None or loader_spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(module)
    return module


def spec_problems(repo: Path) -> list[str]:
    """Check every native-op declaration against the C++ entry point it dispatches to.

    The declared argument names and their order must equal the C++ parameter names, so a facade
    that forwards in declaration order cannot swap two same-typed arguments unnoticed.
    """

    raw, _index = _parsed(repo)
    symbols = frozenset(str(entry["name"]) for entry in raw["symbols"])
    parameters = {
        str(entry["name"]): [str(parameter["name"]) for parameter in entry["parameters"]]
        for entry in raw["symbols"]
    }
    package_root = repo / PACKAGE_PATH
    problems: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        declared = spec_declarations(tree, symbols)
        if not declared:
            continue
        module_name = _module_name(path, package_root)
        module = _load_module(repo, path, module_name)
        for name, symbol in sorted(declared.items()):
            spec = getattr(module, name)
            names = list(getattr(spec, "abi_names", ()))
            if getattr(spec, "symbol", None) != symbol:
                problems.append(f"{module_name}.{name} does not declare {symbol}")
            elif names != parameters[symbol]:
                problems.append(
                    f"{module_name}.{name} arguments {names} differ from the C++ parameters of "
                    f"{symbol} {parameters[symbol]}"
                )
    return problems


def check_invariants(repo: Path) -> list[str]:
    raw, index = _parsed(repo)
    problems = [f"pybind symbol registered more than once: {name}" for name in raw["duplicate_symbols"]]
    symbols = frozenset(str(entry["name"]) for entry in raw["symbols"])
    for entry in raw["symbols"]:
        for label in ("name", "target"):
            if _FORBIDDEN_NAME.search(str(entry[label]).rsplit("::", 1)[-1]):
                problems.append(f"binding {label} carries a generation or work-in-progress marker: {entry[label]}")

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
        for name in sorted(uncalled)
    )
    problems.extend(
        f"native symbol is dispatched from more than one module: {name}"
        for name in sorted(split - KNOWN_SPLIT_OWNER_SYMBOLS)
    )
    problems.extend(
        f"KNOWN_SPLIT_OWNER_SYMBOLS entry is stale (now single-owner or deleted): {name}"
        for name in sorted(KNOWN_SPLIT_OWNER_SYMBOLS - split)
    )
    problems.extend(spec_problems(repo))
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
