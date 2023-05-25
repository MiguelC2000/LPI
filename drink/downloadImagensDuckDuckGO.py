# encoding: utf-8

import DuckDuckGoImages as ddg		#pip install DuckDuckGoImages


filtro = 'person drinking water'
destino = 'C:\\Projeto_LPI\\dataset'

print('Iniciando downloads...')
try:
    ddg.download(filtro, folder=destino, remove_folder=True, parallel=True)
except Exception as e:
    print("type error: ", e)
print('Downloads concluidos...')