import wack
import sys

if len(sys.argv) > 1:
    fname = sys.argv[1]
    try:
        with open(fname, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        asts = []
        for line in lines:
            lexer = wack.Lexer(fname, line)
            tokens, error = lexer.make_tokens()
            if error:
                print(error.as_string())
                sys.exit(1)

            parser = wack.Parser(tokens)
            ast = parser.parse()
            if ast.error:
                print(ast.error.as_string())
                sys.exit(1)

            asts.append(ast.node)

        interpreter = wack.Interpreter()
        context = wack.Context("<program>")
        context.symbol_table = wack.symbol_table

        result = None
        for ast_node in asts:
            res = interpreter.visit(ast_node, context)
            if res.error:
                print(res.error.as_string())
                sys.exit(1)
            result = res.value
            if result:
                print(result)

    except FileNotFoundError:
        print(f"error: {fname} not found")
