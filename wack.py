DIGITS = "0123456789"


class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f"{self.error_name}: {self.details}\n"
        result += f"File {self.pos_start.fn}, line {self.pos_start.ln + 1}"
        return result


class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, "Illegal Character", details)


class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=""):
        super().__init__(pos_start, pos_end, "Invalid Syntax", details)


class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, "Runtime Error", details)
        self.context = context

    def as_string(self):
        result = self.generate_traceback()
        result += f"{self.error_name}: {self.details}"
        return result

    def generate_traceback(self):
        result = ""
        pos = self.pos_start
        ctx = self.context
        while ctx:
            result = (
                f"  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n"
                + result
            )
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return "Traceback (most recent call last):\n" + result


class CharError(Error):
    def __init__(self, pos_start, pos_end, details=""):
        super().__init__(pos_start, pos_end, "expected char", details)


class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == "\n":
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)


TT_INT = "INT"
TT_FLOAT = "FLOAT"
TT_PLUS = "PLUS"
TT_MINUS = "MINUS"
TT_MUL = "MUL"
TT_DIV = "DIV"
TT_LPAREN = "LPAREN"
TT_RPAREN = "RPAREN"
TT_EOF = "EOF"
TT_POW = "POW"
TT_ID = "ID"
TT_KEYWORD = "KEYWORD"
TT_EQ = "EQ"
TT_EE = "EE"
TT_NE = "NE"
TT_GT = "GT"
TT_LT = "LT"
TT_LTE = "LTE"
TT_GTE = "GTE"
TT_MOD = "MOD"

KEYWORDS = ["LET", "AND", "OR", "NOT", "IF", "ELIF", "ELSE", "DO", "FOR", "IN", "SHOW"]


class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value:
            return f"{self.type}:{self.value}"
        return f"{self.type}"


class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = (
            self.text[self.pos.idx] if self.pos.idx < len(self.text) else None
        )

    def make_id(self):
        id_str = ""
        pos_start = self.pos.copy()

        while self.current_char != None and (
            self.current_char.isalpha() or self.current_char.isdigit()
        ):
            id_str += self.current_char
            self.advance()

        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_ID
        return Token(tok_type, id_str, pos_start, self.pos)

    def make_tokens(self):
        tokens = []

        while self.current_char != None:
            if self.current_char in " \t":
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char == "+":
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == "-":
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == "*":
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == "/":
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == "^":
                tokens.append(Token(TT_POW, pos_start=self.pos))
                self.advance()
            elif self.current_char == "(":
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ")":
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == "%":
                tokens.append(Token(TT_MOD, pos_start=self.pos))
                self.advance()
            elif self.current_char == "=":
                tokens.append(self.make_EQ())
            elif self.current_char == "<":
                tokens.append(self.make_LT())
            elif self.current_char == ">":
                tokens.append(self.make_GT())
            elif self.current_char.isalpha():
                tokens.append(self.make_id())
            elif self.current_char == "!":
                tok, error = self.make_NE()
                if error:
                    return [], error
                tokens.append(tok)
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_NE(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == "=":
            self.advance()
            return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None
        self.advance()
        return None, CharError(pos_start, self.pos, "'=' after !")

    def make_EQ(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == "=":
            self.advance()
            return Token(TT_EE, pos_start=pos_start, pos_end=self.pos)
        return Token(TT_EQ, pos_start=pos_start, pos_end=self.pos)

    def make_LT(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == "=":
            self.advance()
            return Token(TT_LTE, pos_start=pos_start, pos_end=self.pos)
        return Token(TT_LT, pos_start=pos_start, pos_end=self.pos)

    def make_GT(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == "=":
            self.advance()
            return Token(TT_GTE, pos_start=pos_start, pos_end=self.pos)
        return Token(TT_GT, pos_start=pos_start, pos_end=self.pos)

    def make_number(self):
        num_str = ""
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + ".":
            if self.current_char == ".":
                if dot_count == 1:
                    break
                dot_count += 1
                num_str += "."
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)


class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f"{self.tok}"


class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f"({self.left_node}, {self.op_tok}, {self.right_node})"


class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f"({self.op_tok}, {self.node})"


class VarAccessNode:
    def __init__(self, name_tok):
        self.name_tok = name_tok
        self.pos_start = self.name_tok.pos_start
        self.pos_end = self.name_tok.pos_end


class VarAssignNode:
    def __init__(self, name_tok, val_node):
        self.name_tok = name_tok
        self.val_node = val_node

        self.pos_start = name_tok.pos_start
        self.pos_end = val_node.pos_end


class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = self.else_case or self.cases[-1][0].pos_end


class ForNode:
    def __init__(self, iterator, length, expr):
        self.iterator = iterator
        self.length = length
        self.expr = expr

        self.pos_start = self.iterator.pos_start
        self.pos_end = self.expr.pos_end


class ShowNode:
    def __init__(self, expr):
        self.expr = expr

        self.pos_start = expr.pos_start
        self.pos_end = expr.pos_end


class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error:
                self.error = res.error
            return res.node

        return res

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def parse(self):
        res = self.expr()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(
                InvalidSyntaxError(
                    self.current_tok.pos_start,
                    self.current_tok.pos_end,
                    "Expected an operation(+,-,*,/)",
                )
            )
        return res

    def if_expr(self):
        res = ParseResult()
        cases = []
        else_case = None

        if not self.current_tok.matches(TT_KEYWORD, "IF"):
            return res.failure(
                InvalidSyntaxError(
                    self.current_tok.pos_start,
                    self.current_tok.pos_end,
                    f"Expected 'IF'",
                )
            )

        self.advance()
        cond = res.register(self.expr())
        if res.error:
            return res

        if not self.current_tok.matches(TT_KEYWORD, "DO"):
            return res.failure(
                InvalidSyntaxError(
                    self.current_tok.pos_start,
                    self.current_tok.pos_end,
                    f"Expected 'DO'",
                )
            )

        self.advance()
        expr = res.register(self.expr())
        if res.error:
            return res
        cases.append((cond, expr))

        while self.current_tok.matches(TT_KEYWORD, "ELIF"):
            self.advance()

            cond = res.register(self.expr())
            if res.error:
                return res
            if not self.current_tok.matches(TT_KEYWORD, "DO"):
                return res.failure(
                    InvalidSyntaxError(
                        self.current_tok.pos_start,
                        self.current_tok.pos_end,
                        f"Expected 'DO'",
                    )
                )

            self.advance()

            expr = res.register(self.expr())
            if res.error:
                return res
            cases.append((cond, expr))

        if self.current_tok.matches(TT_KEYWORD, "ELSE"):
            self.advance()
            else_case = res.register(self.expr())
            if res.error:
                return res
        return res.success(IfNode(cases, else_case))

    def for_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, "FOR"):
            return res.failure(
                InvalidSyntaxError(
                    self.current_tok.pos_start,
                    self.current_tok.pos_end,
                    f"Expected 'FOR'",
                )
            )

        self.advance()
        iterator = res.register(self.expr())
        if res.error:
            return res

        if not self.current_tok.matches(TT_KEYWORD, "IN"):
            return res.failure(
                InvalidSyntaxError(
                    self.current_tok.pos_start,
                    self.current_tok.pos_end,
                    f"Expected 'IN'",
                )
            )

        self.advance()
        length = res.register(self.expr())
        if res.error:
            return res

        if not self.current_tok.matches(TT_KEYWORD, "DO"):
            return res.failure(
                InvalidSyntaxError(
                    self.current_tok.pos_start,
                    self.current_tok.pos_end,
                    f"Expected 'DO'",
                )
            )

        self.advance()
        expr = res.register(self.expr())
        if res.error:
            return res

        return res.success(ForNode(iterator, length, expr))

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryOpNode(tok, factor))

        elif tok.type == TT_ID:
            res.register(self.advance())
            return res.success(VarAccessNode(tok))

        elif tok.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))

        elif tok.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error:
                return res
            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(
                    InvalidSyntaxError(
                        self.current_tok.pos_start,
                        self.current_tok.pos_end,
                        "Expected ')'",
                    )
                )
        elif tok.matches(TT_KEYWORD, "IF"):
            if_expr = res.register(self.if_expr())
            if res.error:
                return res
            return res.success(if_expr)

        elif tok.matches(TT_KEYWORD, "FOR"):
            for_expr = res.register(self.for_expr())
            if res.error:
                return res
            return res.success(for_expr)

        return res.failure(
            InvalidSyntaxError(tok.pos_start, tok.pos_end, "Expected int or float")
        )

    def term(self):
        return self.bin_op(self.pow, (TT_MUL, TT_DIV, TT_MOD))

    def pow(self):
        res = ParseResult()
        left = res.register(self.factor())
        if res.error:
            return res

        if self.current_tok.type == TT_POW:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(self.pow())
            if res.error:
                return res
            left = BinOpNode(left, op_tok, right)

        return res.success(left)

    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def comp_expr(self):
        res = ParseResult()
        if self.current_tok.matches(TT_KEYWORD, "NOT"):
            op_tok = self.current_tok
            self.advance()

            node = res.register(self.comp_expr())
            if res.error:
                return res
            return res.success(UnaryOpNode(op_tok, node))

        node = res.register(
            self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_GTE, TT_LTE))
        )
        if res.error:
            return res
        return res.success(node)

    def expr(self):
        res = ParseResult()

        if self.current_tok.matches(TT_KEYWORD, "SHOW"):
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error:
                return res
            return res.success(ShowNode(expr))

        if self.current_tok.matches(TT_KEYWORD, "LET"):
            res.register(self.advance())
            if self.current_tok.type != TT_ID:
                return res.failure(
                    InvalidSyntaxError(
                        self.current_tok.pos_start,
                        self.current_tok.pos_end,
                        "Expected identifier",
                    )
                )
            name = self.current_tok
            res.register(self.advance())
            if self.current_tok.type != TT_EQ:
                return res.failure(
                    InvalidSyntaxError(
                        self.current_tok.pos_start,
                        self.current_tok.pos_end,
                        "Expected '='",
                    )
                )
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error:
                return res
            return res.success(VarAssignNode(name, expr))

        return self.bin_op(self.comp_expr, ((TT_KEYWORD, "AND"), (TT_KEYWORD, "OR")))

    def bin_op(self, func_a, ops, func_b=None):
        if func_b == None:
            func_b = func_a

        res = ParseResult()
        left = res.register(func_a())
        if res.error:
            return res

        while True:
            if ops and isinstance(ops[0], tuple):
                if self.current_tok.matches(ops[0][0], ops[0][1]):
                    op_tok = self.current_tok
                    res.register(self.advance())
                    right = res.register(func_b())
                    if res.error:
                        return res
                    left = BinOpNode(left, op_tok, right)
                else:
                    break
            else:
                if self.current_tok.type in ops:
                    op_tok = self.current_tok
                    res.register(self.advance())
                    right = res.register(func_b())
                    if res.error:
                        return res
                    left = BinOpNode(left, op_tok, right)
                else:
                    break

        return res.success(left)


class RTResult:
    def __init__(self):
        self.value = None
        self.error = None
        self.show = None

    def register(self, res):
        if res.error:
            self.error = res.error
        return res.value

    def success(self, value, show=False):
        self.value = value
        self.show = show
        return self

    def failure(self, error):
        self.error = error
        return self


class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end, "Division by zero", self.context
                )

            return Number(self.value / other.value).set_context(self.context), None

    def powed_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return Number(1).set_context(self.context), None
            return Number(self.value**other.value).set_context(self.set_context), None

    def get_comp_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(
                self.context
            ), None

    def get_comp_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(
                self.context
            ), None

    def get_comp_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None

    def get_comp_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None

    def get_comp_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(
                self.context
            ), None

    def get_comp_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(
                self.context
            ), None

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(
                self.context
            ), None

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(
                self.context
            ), None

    def notted(self):
        return Number(1 if self.value == 0 else 0).set_context(self.context), None

    def modded_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end, "Division by zero", self.context
                )

            return Number(self.value % other.value).set_context(self.context), None

    def is_true(self):
        return self.value != 0

    def __repr__(self):
        return str(self.value)


class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None


class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None

    def get(self, name):
        val = self.symbols.get(name, None)
        if val == None and self.parent:
            return self.parent.get(name)
        return val

    def set(self, name, val):
        self.symbols[name] = val

    def remove(self, name):
        del self.symbols[name]


class Interpreter:
    def visit(self, node, context):
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f"No visit_{type(node).__name__} method defined")

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value)
            .set_context(context)
            .set_pos(node.pos_start, node.pos_end)
        )

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        if node.op_tok.type == TT_PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == TT_MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TT_MUL:
            result, error = left.multed_by(right)
        elif node.op_tok.type == TT_DIV:
            result, error = left.dived_by(right)
        elif node.op_tok.type == TT_POW:
            result, error = left.powed_by(right)
        elif node.op_tok.type == TT_EE:
            result, error = left.get_comp_eq(right)
        elif node.op_tok.type == TT_NE:
            result, error = left.get_comp_ne(right)
        elif node.op_tok.type == TT_LT:
            result, error = left.get_comp_lt(right)
        elif node.op_tok.type == TT_GT:
            result, error = left.get_comp_gt(right)
        elif node.op_tok.type == TT_LTE:
            result, error = left.get_comp_lte(right)
        elif node.op_tok.type == TT_GTE:
            result, error = left.get_comp_gte(right)
        elif node.op_tok.matches(TT_KEYWORD, "AND"):
            result, error = left.anded_by(right)
        elif node.op_tok.matches(TT_KEYWORD, "OR"):
            result, error = left.ored_by(right)
        elif node.op_tok.type == TT_MOD:
            result, error = left.modded_by(right)

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_IfNode(self, node, context):
        res = RTResult()
        for cond, expr in node.cases:
            cond_val = res.register(self.visit(cond, context))
            if res.error:
                return res

            if cond_val.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.error:
                    return res
                return res.success(expr_value)

        if node.else_case:
            else_value = res.register(self.visit(node.else_case, context))
            if res.error:
                return res
            return res.success(else_value)
        return res.success(None)

    def visit_ForNode(self, node, context):
        res = RTResult()

        var_name = node.iterator.name_tok.value

        length_val = res.register(self.visit(node.length, context))
        if res.error:
            return res

        if not isinstance(length_val, Number):
            return res.failure(
                RTError(
                    node.length.pos_start,
                    node.length.pos_end,
                    "Length must be a number",
                    context,
                )
            )

        results = []
        for i in range(1, int(length_val.value)):
            context.symbol_table.set(var_name, Number(i))

            result = res.register(self.visit(node.expr, context))
            if res.error:
                return res

            results.append(result)

        return res.success(results[-1] if results else Number(0))

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error:
            return res

        error = None

        if node.op_tok.type == TT_MINUS:
            number, error = number.multed_by(Number(-1))
        elif node.op_tok.matches(TT_KEYWORD, "NOT"):
            number, error = number.notted()

        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        name = node.name_tok.value
        val = context.symbol_table.get(name)

        if not val:
            return res.failure(
                RTError(node.pos_start, node.pos_end, f"'{name}' not defined", context)
            )
        return res.success(val)

    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        name = node.name_tok.value
        val = res.register(self.visit(node.val_node, context))
        if res.error:
            return res

        context.symbol_table.set(name, val)
        return res.success(val)

    def visit_ShowNode(self, node, context):
        res = RTResult()
        val = res.register(self.visit(node.expr, context))
        if res.error:
            return res
        return res.success(val, show=True)


symbol_table = SymbolTable()
symbol_table.set("FALSE", Number(0))
symbol_table.set("TRUE", Number(1))


def run(fn, text):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()  # tokenize
    if error:
        return None, error

    parser = Parser(tokens)
    ast = parser.parse()  # convert to ast tree
    if ast.error:
        return None, ast.error

    interpreter = Interpreter()
    context = Context("<program>")  # compile
    context.symbol_table = symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error
