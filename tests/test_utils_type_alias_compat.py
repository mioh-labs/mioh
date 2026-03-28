import ast
from pathlib import Path
import unittest


class UtilsTypeAliasCompatTests(unittest.TestCase):
    def test_utils_module_parses_with_python_311_grammar(self):
        source = Path("lada/utils/__init__.py").read_text(encoding="utf-8")
        ast.parse(source, filename="lada/utils/__init__.py", feature_version=(3, 11))


if __name__ == "__main__":
    unittest.main()
