import requests
import pandas as pd
import sys
import os
output_dir = os.path.join(os.path.dirname(__file__), '../data')

def _get_order_history_address(address, start, end, chunk_size=5000):
    fromTimestamp = pd.Timestamp(start).value // 1000000000
    toTimestamp = pd.Timestamp(end).value // 1000000000
    url = f"https://api.kuru.io/api/v3/{address}/user/order-events"
    offset = 0
    dfs = []
    while True:
        response = requests.get(url, params={'fromTimestamp': fromTimestamp, 'toTimestamp': toTimestamp, 'limit': chunk_size, 'offset': offset})
        res_json = response.json()
        data = []
        for item in res_json['data']['data']:
            event_type = item['eventType']
            event_data = item['eventData']
            row = {'eventType': event_type, 'transactionHash': item['transactionHash'], 'blockTimestamp': item['blockTimestamp']}
            row.update({x: y for x, y in event_data.items() if x not in {
                'owner', 'baseToken', 'quoteToken', 'marketAddress',
                'baseDeposited', 'baseDepositedInUsd', 'basePositionValue', 'basePositionValueInUsd', 'baseFilled', 'baseFilledInUsd'
            }})
            if event_type == 'order-created':
                row['baseValue'] = int(event_data['size']) / int(event_data['sizePrecision'])
                row['baseUsd'] = int(event_data['baseDepositedInUsd']) / (int(event_data['sizePrecision']) * int(event_data['pricePrecision']))
                row['price'] = int(event_data['price']) / int(event_data['pricePrecision'])
            elif event_type == 'order-canceled':
                row['baseValue'] = int(event_data['size']) / int(event_data['sizePrecision'])
                row['baseUsd'] = int(event_data['basePositionValueInUsd']) / (int(event_data['sizePrecision']) * int(event_data['pricePrecision']))
                row['price'] = int(event_data['price']) / int(event_data['pricePrecision'])
            elif event_type == 'trade':
                row['baseValue'] = int(event_data['filledSize']) / int(event_data['sizePrecision'])
                row['baseUsd'] = int(event_data['baseFilledInUsd']) / (int(event_data['sizePrecision']) * int(event_data['pricePrecision']))
                row['price'] = int(event_data['fillPrice']) / (int(event_data['sizePrecision']) * int(event_data['pricePrecision']))
            data.append(row)
        df = pd.DataFrame(data)
        dfs.append(df)
        num_rows_returned = len(res_json['data']['data'])
        if num_rows_returned < chunk_size:
            break
        offset += chunk_size

    result = pd.concat(dfs)
    result['blockTimestamp'] = pd.to_datetime(result['blockTimestamp']).dt.tz_localize(None)
    return result.sort_values('blockTimestamp')

def get_order_history_address(address, start, end, chunk_size=5000):
    dfs = []
    for date in pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(seconds=1), freq='15min'):
        start_time = date
        end_time = date + pd.Timedelta(minutes=15)
        print(f'Getting order history for {address} from {start_time} to {end_time}')
        df = _get_order_history_address(address, start_time, end_time, chunk_size)
        dfs.append(df)
    return pd.concat(dfs)

if __name__ == '__main__':
    t0 = pd.Timestamp.now()
    address = '0x8d0464f646355fb8be1f3107dfbff8547bbf4402'
    start = (t0.floor('1D') - pd.Timedelta(days=1)).strftime('%Y%m%d')
    end = t0.floor('1D').strftime('%Y%m%d')
    if len(sys.argv) > 1:
        date = sys.argv[1]
    if len(sys.argv) > 2:
        address = sys.argv[2]
    df = get_order_history_address(address, start, end)
    print(f'Time taken: {pd.Timestamp.now() - t0}')
    df.to_csv(os.path.join(output_dir, f'order_history_{address}_{start}.csv'), index=False)
