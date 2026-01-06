# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Refresh exported symbols (defined in __all__) in __init__.py files."""

import argparse
import ast
from pathlib import Path


def get_exported_symbols(tree: ast.Module) -> list[str]:
    """Retrieve __all__ list from tree, or None if not found."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return ast.literal_eval(node.value)
    return []


def get_imports(tree: ast.Module) -> list[str]:
    """Return a list of names imported via 'from ... import ...'."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append(alias.asname or alias.name)
    return imports


def create_all_node(symbols: list[str]) -> ast.Assign:
    """Build an __all__ node from a symbol list."""
    node = ast.Assign(
        targets=[ast.Name(id="__all__", ctx=ast.Store())],
        value=ast.List(elts=[ast.Constant(value=s) for s in symbols], ctx=ast.Load()),
    )
    return node


def replace_or_add_all_node(tree: ast.Module, new_all_node: ast.Assign) -> ast.Module:
    """Replace existing __all__ if found, otherwise append a new node."""
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    tree.body[i] = new_all_node
                    return tree
    tree.body.append(new_all_node)
    return tree


def process_file(path: Path):
    """Refresh a file __all__."""
    source = path.read_text()
    tree = ast.parse(source)
    symbols = get_exported_symbols(tree)
    imports = get_imports(tree)
    if symbols == imports:
        return
    all_node = create_all_node(imports)
    tree = replace_or_add_all_node(tree, all_node)
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    path.write_text(source)


def main():
    """Process each __init__.py file given as argument."""
    parser = argparse.ArgumentParser(
        description="Refresh __all__ in __init__.py files."
    )
    parser.add_argument(
        "files", nargs="+", type=Path, help="Python source files to process"
    )
    args = parser.parse_args()
    for file_path in args.files:
        if file_path.name == "__init__.py":
            process_file(file_path)


if __name__ == "__main__":
    main()
