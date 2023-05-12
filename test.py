from myparser import TatsuParser, TatsuSemantics

parser_tatsu = TatsuParser()
ast = parser_tatsu.parse("person bottle person bottle person", start='start')
semantics = TatsuSemantics()
output = semantics.expression(ast)
print(output)
