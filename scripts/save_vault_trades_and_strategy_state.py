import pandas as pd
import matplotlib.pyplot as plt
import sys

import dotenv
dotenv.load_dotenv()

import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.kuru import client, dune_client, get_kuru_vault_token_supply, VAULT_TOKEN_ADDRESSES

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data')

def get_vault_trades(start_date, end_date, market='MONUSDC', plot_filename=None):
    start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d %H:%M:%S')
    end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d %H:%M:%S')
    df = client.query(f"SELECT * FROM trade WHERE time >= '{start_str}'::timestamp AND time < '{end_str}'::timestamp AND market = '{market}'").to_pandas()
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

    chunks = []
    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + pd.Timedelta(hours=1), end_ts)
        chunk_start_str = chunk_start.strftime('%Y-%m-%d %H:%M:%S')
        chunk_end_str = chunk_end.strftime('%Y-%m-%d %H:%M:%S')
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
        chunk_start = chunk_end

    df = pd.concat(chunks, ignore_index=True) if chunks else chunks[0]
    if dune_client is not None:
        vault_token_supply = get_kuru_vault_token_supply(VAULT_TOKEN_ADDRESSES[market])

    df['time'] = pd.to_datetime(df['time'])
    if dune_client is not None:
        df = pd.merge_asof(df, vault_token_supply[['time', 'total_supply']], on='time')
        df['benchmark'] = np.sqrt(df['fair_value'] / df['fair_value'].iloc[0])
        df['perf'] = (df['base_balance'] * df['fair_value'] + df['quote_balance']) / df['total_supply']
        df['perf'] = df['perf'] / df['perf'].iloc[0]
    return df.set_index('time').sort_index()

def _load_existing(path):
    """Return (df indexed by time, max_ts) or (None, None) if file missing/empty."""
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    if df.empty or 'time' not in df.columns:
        return None, None
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    return df, df.index.max()


def _merge_and_write(existing, new, path, label):
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new
    combined.to_csv(path)
    print(f"Saved {len(combined)} {label} rows to {path}")


def save_data(start, end, market, force=False):
    date_str = pd.Timestamp(start).strftime('%Y%m%d')
    trades_output          = os.path.join(OUTPUT_DIR, f'{market}_vault_trades_{date_str}.csv')
    strategy_state_output  = os.path.join(OUTPUT_DIR, f'{market}_vault_strategy_state_{date_str}.csv')

    # ── Trades ──────────────────────────────────────────────────────────────
    existing_trades, trades_last = (None, None) if force else _load_existing(trades_output)
    trades_start = trades_last if trades_last is not None else start
    if trades_last is not None:
        print(f"Trades: resuming from {trades_start} (existing file has data through that point)")
    if trades_start < end:
        new_trades = get_vault_trades(trades_start, end, market=market)
        _merge_and_write(existing_trades, new_trades, trades_output, "vault trade")
    else:
        print(f"Trades: already up to date through {trades_start}, skipping fetch")

    # ── Strategy state ──────────────────────────────────────────────────────
    existing_ss, ss_last = (None, None) if force else _load_existing(strategy_state_output)
    ss_start = ss_last if ss_last is not None else start
    if ss_last is not None:
        print(f"Strategy state: resuming from {ss_start}")
    if ss_start < end:
        df = get_vault_strategy_state(ss_start, end, market=market)
        new_ss = df.resample('0.4S', closed='right', label='right').last()
        new_ss['tvl'] = new_ss['base_balance'] * new_ss['fair_value'] + new_ss['quote_balance']
        new_ss['pos'] = (2 * new_ss['base_balance'] * new_ss['fair_value'] / new_ss['tvl'] - 1)
        _merge_and_write(existing_ss, new_ss, strategy_state_output, "strategy state")
    else:
        print(f"Strategy state: already up to date through {ss_start}, skipping fetch")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fetch vault trades and strategy state.')
    parser.add_argument('--date', metavar='YYYYMMDD', help='Fetch data for a full day, e.g. 20250411')
    parser.add_argument('--from', dest='from_ts', metavar='TIMESTAMP', help='Start timestamp, e.g. "2025-04-11 08:00:00"')
    parser.add_argument('--to', dest='to_ts', metavar='TIMESTAMP', help='End timestamp, e.g. "2025-04-11 16:00:00"')
    parser.add_argument('--market', default='MONUSDC', help='Market (default: MONUSDC)')
    parser.add_argument('--force', action='store_true',
                        help='Refetch from scratch even if the output CSVs already exist.')
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

    save_data(start, end, args.market, force=args.force)
