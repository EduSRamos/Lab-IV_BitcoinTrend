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
price = client.get_spot_price(currency_pair = 'BTC-USD')
print(price)