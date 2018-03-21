'''
A API do Coinbase possui alguns sistemas de segurança e 
autenticação que são necessários para utiliza-la.
O primeiro passo para utiliza-la é criar uma conta no
www.coinbase.com
Eu liberei o acesso da minha para todos os IP's, então
vou deixa-la hard-coded aqui e podem usa-la. É bom
ter isso em mente apenas para caso estejam testando
em casa e por algum motivo tome erro de autenticação.
'''

from coinbase.wallet.client import Client
from coinbase.wallet.model import APIObject

api_key = 'R6NS3FRYnJXKJmI7' 
api_secret = '2b4Pdpji2iYZrdBMalPEKerWKs8FCFh6'

#Fazer autenticação
client = Client(api_key, api_secret)

#Pegar preço de mercado - Argumento date opcional
#pode ser utilizada para pegar o preço em um dia específico
#o argumento é no formato date = 'YYYY-MM-DD'
price = client.get_spot_price(currency_pair = 'BTC-USD', date = '2018-01-01T07:10:40')
print(price)

#Pegar o histórico de preços - Esta função pega o histórico de preços.
#O único parâmetro dela é  o period que pode ter os seguintes valores:
#hour - Pega o histórico da última hora (361 pontos)
#day - Pega o histórico das últimas 24h (360 pontos)
#week - Pega o histórico da última semana (316 pontos)
#month - Pega o histórico do último mês (360 pontos)
#year - Pega o histórico do último ano (365 pontos)
#all - Pegar o histórico do último ano(não entendi, mas pega mais pontos)
historic = client.get_historic_prices(currency_pair = 'BTC-USD', period = 'hour')

print(historic)
print(len(historic["prices"]))
