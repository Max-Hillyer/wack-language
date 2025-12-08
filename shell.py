import wack
import sys

if len(sys.argv) > 1:
    fname = sys.argv[1]
    try:
        with open(fname, "r") as f:
            text = f.readlines()
        for i in text:
            res, err = wack.run(fname, i)
            if err:
                print(err.as_string())
            else:
                print(res)
    except FileNotFoundError:
        print(f"error: {fname} not found")
else:
    while True:
        text = input("wack > ")
        if text == "cutitout":
            break
        result, error = wack.run("<stdin>", text)

        if error:
            print(error.as_string())
        else:
            print(result)
