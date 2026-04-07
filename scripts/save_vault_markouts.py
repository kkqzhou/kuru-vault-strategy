import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

import dotenv
dotenv.load_dotenv()

import sys
sys.path.append('..')

from lib.kuru import get_kuru_vault_token_supply, VAULT_TOKEN_ADDRESSES
from lib.trading import compute_markouts, print_trading_report

from influxdb_client_3 import InfluxDBClient3

pd.options.mode.chained_assignment = None

INFLUXDB_URL = os.getenv('KURU_STRATEGY_INFLUXDB_URL')  
INFLUXDB_TOKEN = os.getenv('KURU_STRATEGY_INFLUXDB_TOKEN')
DATABASE = os.getenv('KURU_STRATEGY_INFLUXDB_DATABASE')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data')

client = InfluxDBClient3(host=INFLUXDB_URL, token=INFLUXDB_TOKEN, database=DATABASE)

# print(client.query('SHOW TABLES').to_pandas())
"""
   table_catalog        table_schema                           table_name  table_type
0         public                 iox                     backrun_gas_info  BASE TABLE
1         public                 iox            backrun_operator_balances  BASE TABLE
2         public                 iox                      backrun_tx_info  BASE TABLE
3         public                 iox                         confirmed_tx  BASE TABLE
4         public                 iox                              latency  BASE TABLE
5         public                 iox                           new_quotes  BASE TABLE
6         public                 iox                       order_canceled  BASE TABLE
7         public                 iox                         order_posted  BASE TABLE
8         public                 iox                        orderbook_bbo  BASE TABLE
9         public                 iox      position_bias_parameter_changes  BASE TABLE
10        public                 iox                        revert_reason  BASE TABLE
11        public                 iox                              reverts  BASE TABLE
12        public                 iox                            sent_txns  BASE TABLE
13        public                 iox               skew_parameter_changes  BASE TABLE
14        public                 iox                       strategy_state  BASE TABLE
15        public                 iox              tenderly_simulated_txns  BASE TABLE
16        public                 iox                                trade  BASE TABLE
17        public              system                      distinct_caches  BASE TABLE
18        public              system                      influxdb_schema  BASE TABLE
19        public              system                          last_caches  BASE TABLE
20        public              system                        parquet_files  BASE TABLE
21        public              system               processing_engine_logs  BASE TABLE
22        public              system  processing_engine_trigger_arguments  BASE TABLE
23        public              system           processing_engine_triggers  BASE TABLE
24        public              system                              queries  BASE TABLE
25        public  information_schema                               tables        VIEW
26        public  information_schema                                views        VIEW
27        public  information_schema                              columns        VIEW
28        public  information_schema                          df_settings        VIEW
29        public  information_schema                             schemata        VIEW
30        public  information_schema                             routines        VIEW
31        public  information_schema                           parameters        VIEW
"""

def get_vault_markouts(date, market='MONUSDC', report=False):
    date_str = date.strftime('%Y%m%d')
    trades = pd.read_csv(os.path.join(OUTPUT_DIR, f'{market}_vault_trades_{date_str}.csv'))
    strategy_state = pd.read_csv(os.path.join(OUTPUT_DIR, f'{market}_vault_strategy_state_{date_str}.csv'))
    strategy_state['time'] = pd.to_datetime(strategy_state['time'])
    strategy_state = strategy_state.set_index('time')

    # compute and export markouts
    mos_fair_value = compute_markouts(trades, strategy_state['fair_value'], lags=[0, 1, 5, 15, 60])
    mos_fair_value = pd.merge_asof(mos_fair_value, strategy_state[['pos_adj_bps', 'pos', 'tvl', 'fair_value', 'dislocation_bps', 'skew_bps']], on='time')
    if report:
        print_trading_report(
            mos_fair_value,
            markout_col='mo_fair_value_5s',
            aux_print_cols=['time', 'fair_value', 'is_bid', 'price', 'size_base', 'size_usd', 'mo_fair_value_0s', 'mo_fair_value_5s', 'pos', 'pos_adj_bps', 'skew_bps'],
            time_index=pd.to_datetime(trades['time']).dt.tz_localize(None),
            size_col_name='size_base',
            fee_col_name=None
        )
    return mos_fair_value


if __name__  == '__main__':
    if len(sys.argv) == 1:
        print('Usage: python vault_trading_summary.py <yyyymmdd> [market]')
        print('If market is not provided, it will default to MONUSDC')
        sys.exit(1)
    elif len(sys.argv) == 2:
        date = pd.Timestamp(sys.argv[1])
        date_str = date.strftime('%Y%m%d')
        mos_fair_value = get_vault_markouts(date, market='MONUSDC', report=True)
        mos_fair_value.to_csv(f'{OUTPUT_DIR}/MONUSDC_vault_markouts_{date_str}.csv', index=False)
    elif len(sys.argv) == 3:
        date = pd.Timestamp(sys.argv[1])
        date_str = date.strftime('%Y%m%d')
        market = sys.argv[2]
        mos_fair_value = get_vault_markouts(date, market=market)
        print(mos_fair_value['mo_fair_value_5s'].describe())
        mos_fair_value.to_csv(f'{OUTPUT_DIR}/{market}_vault_markouts_{date_str}.csv', index=False)
