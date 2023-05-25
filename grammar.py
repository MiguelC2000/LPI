import tatsu.tool as parser


def create_grammar():
    # model = parser.to_python_model(grammar)
    # model = parser.compile(grammar,asmodel=True)
    f = open("Gramatica.txt", "r")
    grammar = ''.join(f.readlines())
    print(grammar)
    f.close()
    model = parser.to_python_sourcecode(grammar)
    file = open("newmyparser.py", "w")
    print(model)
    file.write(model)



if __name__ == '__main__':
    create_grammar()
