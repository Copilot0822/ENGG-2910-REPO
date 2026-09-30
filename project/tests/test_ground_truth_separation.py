import ast
from pathlib import Path

import inference.full_library
import inference.infer


def test_discovery_module_has_no_simulator_or_answer_key_dependency():
    for module in (inference.infer, inference.full_library):
        path = Path(module.__file__)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not any(
            name == "simulator" or name.startswith("simulator.") for name in imported
        ), path
        assert "ground_truth.json" not in source, path
        assert "signal_catalog.csv" not in source, path
        assert "canboat.json" not in source, path
        assert "J1939_2024" not in source, path
        assert "ground_truth" not in source, path
