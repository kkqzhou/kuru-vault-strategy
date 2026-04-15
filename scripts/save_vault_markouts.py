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
from lib.trading import compute_markouts, print_trading_report, print_vault_performance_report

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

def get_vault_markouts(date, market='MONUSDC', report=False, start=None, end=None):
    date_str = date.strftime('%Y%m%d')
    trades = pd.read_csv(os.path.join(OUTPUT_DIR, f'{market}_vault_trades_{date_str}.csv'))
    strategy_state = pd.read_csv(os.path.join(OUTPUT_DIR, f'{market}_vault_strategy_state_{date_str}.csv'))
    strategy_state['time'] = pd.to_datetime(strategy_state['time'])
    strategy_state = strategy_state.set_index('time')

    if start is not None and end is not None:
        trades['time'] = pd.to_datetime(trades['time'])
        trades = trades[(trades['time'] >= start) & (trades['time'] < end)]
        strategy_state = strategy_state[(strategy_state.index >= start) & (strategy_state.index < end)]

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
        print_vault_performance_report(strategy_state, mos_fair_value)
    return mos_fair_value


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute vault markout analysis.')
    parser.add_argument('--date', metavar='YYYYMMDD', help='Run analysis for a full day, e.g. 20250411')
    parser.add_argument('--from', dest='from_ts', metavar='TIMESTAMP', help='Start timestamp, e.g. "2025-04-11 08:00:00"')
    parser.add_argument('--to', dest='to_ts', metavar='TIMESTAMP', help='End timestamp, e.g. "2025-04-11 16:00:00"')
    parser.add_argument('--market', default='MONUSDC', help='Market (default: MONUSDC)')
    args = parser.parse_args()

    if args.date and (args.from_ts or args.to_ts):
        parser.error('Provide either --date or --from/--to, not both.')
    if not args.date and not (args.from_ts and args.to_ts):
        parser.error('Provide either --date or both --from and --to.')

    if args.date:
        start = pd.Timestamp(args.date)
        end = start + pd.Timedelta(days=1)
        filter_start, filter_end = None, None
    else:
        start = pd.Timestamp(args.from_ts)
        end = pd.Timestamp(args.to_ts)
        filter_start, filter_end = start, end

    date_str = start.strftime('%Y%m%d')
    mos_fair_value = get_vault_markouts(start, market=args.market, report=True, start=filter_start, end=filter_end)
    mos_fair_value.to_csv(f'{OUTPUT_DIR}/{args.market}_vault_markouts_{date_str}.csv', index=False)
