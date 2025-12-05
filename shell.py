import wack

while True:
    text = input("wack > ")
    if text == "cutitout":
        break
    result, error = wack.run("<stdin>", text)

    if error:
        print(error.as_string())
    else:
        print(result)
