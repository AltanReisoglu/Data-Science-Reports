"""Güvenli bir aritmetik hesaplayıcı (Faz 4, tool sayısını artırma isteği,
2026-08-30). Bilerek `eval()` KULLANMIYOR — LLM'in ürettiği bir ifadeyi
doğrudan `eval` etmek keyfi kod çalıştırma riski taşır. Bunun yerine, yalnızca
temel aritmetik operatörlere izin veren kısıtlı bir AST yürütücüsü kullanır —
`import`, fonksiyon çağrısı, attribute erişimi gibi hiçbir şey desteklenmiyor,
sadece sayılar ve `+ - * / ** % //`.
"""

from __future__ import annotations

import ast
import operator

_ALLOWED_BINOPS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS: dict[type, object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"desteklenmeyen ifade parçası: {ast.dump(node)}")


def calculate(expression: str) -> dict:
    """`expression`'ı (yalnızca +-*/**%//, parantez, sayılar) güvenle hesaplar."""
    try:
        tree = ast.parse(expression, mode="eval")
        return {"expression": expression, "result": _eval_node(tree.body)}
    except Exception as exc:  # noqa: BLE001 - kullanıcı/LLM girdisi, her hata mesaj olarak dönmeli
        return {"expression": expression, "error": str(exc)}
