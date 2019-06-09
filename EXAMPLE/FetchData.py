import time
from UTIL import FileIO
from DATA import API


config_path = 'CONFIG\config_data.yml'
config      = FileIO.read_yaml(config_path)
attr        = ['date', 'close']
ticker_list = ['GOOG', 'GB']
api         = API.Tiingo(config)

start   = time.time()
price_1 = api.fetch(ticker_list, attr)
end     = time.time()
print('Normal processing time: {time}s.'.format(time=end-start))

config['DataAPIFetchMethod'] = 'async'
start   = time.time()
price_2 = api.fetch(ticker_list, attr)
end     = time.time()
print('Asynchronous processing time: {time}s.'.format(time=end-start))


FileIO.save_csv(price_1['GOOG'], 'GOOG', 'STATICS/PRICE')
FileIO.save_csv(price_2['FB'  ], 'FB'  , 'STATICS/PRICE')
