import requests
stocks = {}
SYMBOL = "AAPL"
params = {
    'access_key': '5964fe2bd21cfa47d5eb1b0930e52922',
    'date_from': '2017-07-12',
    'date_to': '2020-08-20',
    'symbols': SYMBOL,
    'limit': 1000,
    'sort': "ASC"
}
stocks[SYMBOL] = []

api_result = requests.get('http://api.marketstack.com/v1/eod', params)

api_response = api_result.json()
print(api_response)
for data in api_response['data']:
    stocks[SYMBOL]
# for stock_data in api_response['data']:
#     print(u'Ticker %s has a day high of  %s on %s' % (
#       stock_data['symbol']
#       stock_data['high']
#       stock_data['date']
#     ))ate']
# ))