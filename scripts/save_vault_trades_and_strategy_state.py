import pandas as pd
import matplotlib.pyplot as plt
import sys

import dotenv
dotenv.load_dotenv()

import numpy as np
import os
import time
import json
sys.path.append('..')
from lib.kuru import client, dune_client, get_kuru_vault_token_supply, VAULT_TOKEN_ADDRESSES

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data')

# region agent log
_DEBUG_LOG = os.path.join(os.path.dirname(__file__), '../.cursor/debug-bd6fa6.log')
def _dlog(msg, data=None, hypothesis_id=None, run_id='run1'):
    entry = {'sessionId': 'bd6fa6', 'runId': run_id, 'hypothesisId': hypothesis_id,
             'location': 'save_vault_trades_and_strategy_state.py', 'message': msg,
             'data': data or {}, 'timestamp': int(time.time() * 1000)}
    with open(_DEBUG_LOG, 'a') as _f:
        _f.write(json.dumps(entry) + '\n')
# endregion

def get_vault_trades(start_date, end_date, market='MONUSDC', plot_filename=None):
    start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d %H:%M:%S')
    end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d %H:%M:%S')
    # region agent log
    _t0 = time.time()
    _dlog('trades query start', {'start': start_str, 'end': end_str, 'market': market}, hypothesis_id='H-A H-B')
    # endregion
    df = client.query(f"SELECT * FROM trade WHERE time >= '{start_str}'::timestamp AND time < '{end_str}'::timestamp AND market = '{market}'").to_pandas()
    # region agent log
    _dlog('trades query done', {'elapsed_s': round(time.time()-_t0,2), 'rows': len(df)}, hypothesis_id='H-A H-B')
    # endregion
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    df['is_bid'] = df['is_bid'].map({'true': True, 'false': False})
    df['total_cost'] = ((df['is_bid'] * 2 - 1) * df['price'] * df['size_base']).cumsum()
    df['total_units'] = ((df['is_bid'] * 2 - 1) * df['size_base']).cumsum()
    df['rolling_pnl'] = df['total_units'] * df['price'] - df['total_cost']
    if plot_filename is not None:
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(df['rolling_pnl'])
        fig.savefig(plot_filename)
        plt.close(fig)
    return df

def get_vault_strategy_state(start_date, end_date, market='MONUSDC'):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    # region agent log
    _t0 = time.time()
    _dlog('strategy_state query start (chunked)', {'start': str(start_ts), 'end': str(end_ts), 'market': market, 'chunk_hours': 1}, hypothesis_id='H-A H-B', run_id='post-fix')
    # endregion

    chunks = []
    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + pd.Timedelta(hours=1), end_ts)
        chunk_start_str = chunk_start.strftime('%Y-%m-%d %H:%M:%S')
        chunk_end_str = chunk_end.strftime('%Y-%m-%d %H:%M:%S')
        _chunk_t0 = time.time()
        chunk_df = client.query(
            f"""SELECT
                time,
                bid_0 as best_bid,
                ask_0 as best_ask,
                bid_fair_value,
                (bid_fair_value + ask_fair_value) * 0.5 as fair_value,
                ask_fair_value,
                skew_bps,
                position_adjustment_bps as pos_adj_bps,
                finalized_base_balance as base_balance,
                finalized_quote_balance as quote_balance,
                dislocation_bps
            FROM strategy_state where time >= '{chunk_start_str}'::timestamp and time < '{chunk_end_str}'::timestamp and market = '{market}'
            ORDER BY time
            """
        ).to_pandas()
        chunks.append(chunk_df)
        # region agent log
        _dlog('strategy_state chunk done', {'chunk_start': chunk_start_str, 'chunk_end': chunk_end_str, 'elapsed_s': round(time.time()-_chunk_t0,2), 'rows': len(chunk_df), 'chunk_num': len(chunks)}, hypothesis_id='H-A H-B', run_id='post-fix')
        # endregion
        chunk_start = chunk_end

    df = pd.concat(chunks, ignore_index=True) if chunks else chunks[0]
    # region agent log
    _dlog('strategy_state query done (chunked)', {'elapsed_s': round(time.time()-_t0,2), 'total_rows': len(df), 'chunks': len(chunks)}, hypothesis_id='H-A H-B', run_id='post-fix')
    # endregion
    if dune_client is not None:
        vault_token_supply = get_kuru_vault_token_supply(VAULT_TOKEN_ADDRESSES[market])

    df['time'] = pd.to_datetime(df['time'])
    if dune_client is not None:
        df = pd.merge_asof(df, vault_token_supply[['time', 'total_supply']], on='time')
        df['benchmark'] = np.sqrt(df['fair_value'] / df['fair_value'].iloc[0])
        df['perf'] = (df['base_balance'] * df['fair_value'] + df['quote_balance']) / df['total_supply']
        df['perf'] = df['perf'] / df['perf'].iloc[0]
    return df.set_index('time').sort_index()

def save_data(start, end, market):
    date_str = pd.Timestamp(start).strftime('%Y%m%d')

    trades_df = get_vault_trades(start, end, market=market)
    trades_output = os.path.join(OUTPUT_DIR, f'{market}_vault_trades_{date_str}.csv')
    trades_df.to_csv(trades_output)
    print(f"Saved vault trades to {trades_output}")

    df = get_vault_strategy_state(start, end, market=market)
    strategy_state_df = df.resample('0.4S', closed='right', label='right').last()
    strategy_state_df['tvl'] = strategy_state_df['base_balance'] * strategy_state_df['fair_value'] + strategy_state_df['quote_balance']
    strategy_state_df['pos'] = (2 * strategy_state_df['base_balance'] * strategy_state_df['fair_value'] / strategy_state_df['tvl'] - 1)

    strategy_state_output = os.path.join(OUTPUT_DIR, f'{market}_vault_strategy_state_{date_str}.csv')
    strategy_state_df.to_csv(strategy_state_output)
    print(f"Saved vault strategy state to {strategy_state_output}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fetch vault trades and strategy state.')
    parser.add_argument('--date', metavar='YYYYMMDD', help='Fetch data for a full day, e.g. 20250411')
    parser.add_argument('--from', dest='from_ts', metavar='TIMESTAMP', help='Start timestamp, e.g. "2025-04-11 08:00:00"')
    parser.add_argument('--to', dest='to_ts', metavar='TIMESTAMP', help='End timestamp, e.g. "2025-04-11 16:00:00"')
    parser.add_argument('--market', default='MONUSDC', help='Market (default: MONUSDC)')
    args = parser.parse_args()

    if args.date and (args.from_ts or args.to_ts):
        parser.error('Provide either --date or --from/--to, not both.')
    if not args.date and not (args.from_ts and args.to_ts):
        # default to yesterday
        start = pd.Timestamp.now().floor('1D') - pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=1)
    elif args.date:
        start = pd.Timestamp(args.date)
        end = start + pd.Timedelta(days=1)
    else:
        if not args.from_ts or not args.to_ts:
            parser.error('Both --from and --to are required when not using --date.')
        start = pd.Timestamp(args.from_ts)
        end = pd.Timestamp(args.to_ts)

    # region agent log
    _dlog('script start', {'start': str(start), 'end': str(end), 'market': args.market}, hypothesis_id='H-E')
    # endregion
    try:
        save_data(start, end, args.market)
    except Exception as _e:
        # region agent log
        _dlog('script error', {'error_type': type(_e).__name__, 'error': str(_e)[:300]}, hypothesis_id='H-B H-E')
        # endregion
        raise
