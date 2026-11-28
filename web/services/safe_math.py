# Безопасный AST-парсер математических выражений для mpmath (без eval/exec)
import ast
import operator
from typing import Any

import mpmath as mp
import numpy as np

MAX_EXPR_LEN = 160
MAX_AST_DEPTH = 12
MAX_POWER_ABS = 20.0
MIN_LENGTH = 100
MAX_LENGTH = 50_000


class SafeMathError(ValueError):
    """Ошибка разбора или вычисления пользовательской формулы."""


def _const_map() -> dict[str, Any]:
    return {
        "pi": mp.pi,
        "e": mp.e,
        "phi": (mp.mpf(1) + mp.sqrt(5)) / 2,
        "euler": mp.euler,
        "catalan": mp.catalan,
        "khinchin": mp.khinchin,
    }


def _func_map() -> dict[str, Any]:
    return {
        "sqrt": mp.sqrt,
        "sin": mp.sin,
        "cos": mp.cos,
        "tan": mp.tan,
        "exp": mp.exp,
        "log": mp.log,
        "ln": mp.log,
        "abs": mp.fabs,
        "frac": lambda x: x - mp.floor(x),
    }


_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_ast_node(node: ast.AST, depth: int, consts: dict[str, Any], funcs: dict[str, Any]) -> Any:
    if depth > MAX_AST_DEPTH:
        raise SafeMathError(f"Превышена допустимая глубина выражения ({MAX_AST_DEPTH})")

    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, depth + 1, consts, funcs)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise SafeMathError("Разрешены только вещественные числовые константы")
        if abs(float(node.value)) > 1e6:
            raise SafeMathError("Числовая константа превышает допустимый предел 1e6")
        return mp.mpf(str(node.value))

    if isinstance(node, ast.Name):
        if node.id not in consts:
            raise SafeMathError(f"Неразрешённый идентификатор: '{node.id}'")
        return consts[node.id]

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise SafeMathError("Неподдерживаемая унарная операция")
        return op_fn(_eval_ast_node(node.operand, depth + 1, consts, funcs))

    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, depth + 1, consts, funcs)
        right = _eval_ast_node(node.right, depth + 1, consts, funcs)
        if isinstance(node.op, ast.Pow):
            if abs(float(right)) > MAX_POWER_ABS:
                raise SafeMathError(f"Показатель степени ограничен |p| <= {int(MAX_POWER_ABS)}")
            if left == 0 and right < 0:
                raise SafeMathError("Возведение нуля в отрицательную степень")
            return left**right
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise SafeMathError("Неподдерживаемая бинарная операция")
        if isinstance(node.op, ast.Div) and right == 0:
            raise SafeMathError("Деление на ноль")
        return op_fn(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise SafeMathError("Разрешены только простые вызовы функций белого списка")
        fn = funcs.get(node.func.id)
        if fn is None:
            raise SafeMathError(f"Функция '{node.func.id}' не входит в белый список")
        if len(node.args) != 1:
            raise SafeMathError(f"Функция '{node.func.id}' принимает ровно 1 аргумент")
        arg = _eval_ast_node(node.args[0], depth + 1, consts, funcs)
        if node.func.id == "sqrt" and arg < 0:
            raise SafeMathError("Квадратный корень из отрицательного числа не поддерживается")
        if node.func.id in ("log", "ln") and arg <= 0:
            raise SafeMathError("Логарифм определён только для положительных значений")
        return fn(arg)

    raise SafeMathError(f"Запрещённый узел синтаксиса: {type(node).__name__}")


def validate_formula_ast(expression: str) -> ast.Expression:
    """Проверяет выражение AST-парсером до вычисления (отсекает любые инъекции)."""
    cleaned = expression.strip()
    if not cleaned or len(cleaned) > MAX_EXPR_LEN:
        raise SafeMathError(f"Длина формулы должна быть от 1 до {MAX_EXPR_LEN} символов")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise SafeMathError(f"Синтаксическая ошибка в выражении: {exc.msg}") from exc
    return tree


def evaluate_formula_digits(expression: str, length: int, radix: int = 10) -> np.ndarray:
    """Вычисляет дробные цифры выражения mpmath в Base-2 / Base-10 / Base-16."""
    if radix not in (2, 10, 16):
        raise SafeMathError(f"Для mpmath поддерживаются только radix in {{2, 10, 16}}, получено {radix}")
    if not (MIN_LENGTH <= int(length) <= MAX_LENGTH):
        raise SafeMathError(f"Длина length должна быть в [{MIN_LENGTH}, {MAX_LENGTH}]")

    n = int(length)
    tree = validate_formula_ast(expression)
    consts = _const_map()
    funcs = _func_map()

    if radix == 10:
        with mp.workdps(n + 48):
            val = _eval_ast_node(tree, 0, consts, funcs)
            if isinstance(val, mp.mpc) or not mp.isfinite(val):
                raise SafeMathError("Выражение дало бесконечное или комплексное значение")
            frac = abs(val) - mp.floor(abs(val))
            if frac == 0:
                return np.zeros(n, dtype=np.uint8)
            # Сдвиг +1 гарантирует отсутствие научной записи при малых дробях (0.000...)
            s = mp.nstr(mp.mpf(1) + frac, n + 20, strip_zeros=False)
        if "." not in s:
            raise SafeMathError("Не удалось извлечь дробную часть")
        frac_str = s.split(".", 1)[1][:n]
        if len(frac_str) < n:
            frac_str = frac_str.ljust(n, "0")
        return (np.frombuffer(frac_str.encode("ascii"), dtype=np.uint8) - ord("0")).copy()

    bits_per_digit = 1 if radix == 2 else 4
    with mp.workprec(n * bits_per_digit + 128):
        val = _eval_ast_node(tree, 0, consts, funcs)
        if isinstance(val, mp.mpc) or not mp.isfinite(val):
            raise SafeMathError("Выражение дало бесконечное или комплексное значение")
        frac = abs(val) - mp.floor(abs(val))
        if frac == 0:
            return np.zeros(n, dtype=np.uint8)
        scaled = int(mp.ldexp(frac, n * bits_per_digit))

    if radix == 2:
        raw_str = f"{scaled:0{n}b}"[-n:]
        return (np.frombuffer(raw_str.encode("ascii"), dtype=np.uint8) - ord("0")).copy()

    raw_str = f"{scaled:0{n}x}"[-n:]
    raw_bytes = np.frombuffer(raw_str.encode("ascii"), dtype=np.uint8)
    return np.where(raw_bytes <= ord("9"), raw_bytes - ord("0"), raw_bytes - ord("a") + 10).astype(np.uint8)
