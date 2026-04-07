import numpy as np
import pandas as pd
import psycopg
import os
from influxdb_client_3 import InfluxDBClient3
from dune_client.client import DuneClient

import dotenv
dotenv.load_dotenv()

pd.options.mode.chained_assignment = None

INFLUXDB_URL = os.getenv('KURU_STRATEGY_INFLUXDB_URL')
INFLUXDB_TOKEN = os.getenv('KURU_STRATEGY_INFLUXDB_TOKEN')
DATABASE = os.getenv('KURU_STRATEGY_INFLUXDB_DATABASE')

client = InfluxDBClient3(host=INFLUXDB_URL, token=INFLUXDB_TOKEN, database=DATABASE)
dune_client = DuneClient.from_env()

VAULT_TOKEN_ADDRESSES = {
    'MONAUSD': '0x4869A4C7657cEf5E5496C9cE56DDe4CD593e4923',
    'MONUSDC': '0xd0F8A6422CcdD812f29D8FB75CF5FCd41483BaDc'
}

def dune_query(sql: str) -> pd.DataFrame:
    result = dune_client.run_sql(sql)
    return pd.DataFrame(result.result.rows)


def get_strategy_state(start_date, end_date, market='MONAUSD', columns=None, include_all_columns=False):
    start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d %H:%M:%S')
    end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d %H:%M:%S')

    # require time to be returned as a column
    if columns is not None and 'time' not in columns:
        columns = ['time'] + columns

    columns_clause = '*' if columns is None else ', '.join(columns)
    df = client.query(
        f"""
        SELECT {columns_clause} from strategy_state
        where time >= '{start_str}'::timestamp and time < '{end_str}'::timestamp and market = '{market}'
        order by time
        """
    ).to_pandas()
    df['time'] = pd.to_datetime(df['time'])
    if not include_all_columns:
        relevant_cols = [x for x in df.columns if x not in {f'ask_{i}' for i in range(1, 100)}.union({f'bid_{i}' for i in range(1, 100)})]
    else:
        relevant_cols = list(df.columns)
    return df[relevant_cols].set_index('time')


def get_kuru_vault_token_supply(vault_token_address):
    sql = f"""
    WITH supply_changes AS (
    SELECT 
        evt_block_time as block_time,
        evt_block_number as block_number,
        evt_tx_hash as tx_hash,
        contract_address as token,
        CASE 
            WHEN "from" = 0x0000000000000000000000000000000000000000 THEN value / 1e18  -- Mint
            WHEN "to" = 0x0000000000000000000000000000000000000000 THEN -value / 1e18   -- Burn
        END AS supply_change
    FROM erc20_monad.evt_transfer
    WHERE contract_address = {vault_token_address}
    AND ("from" = 0x0000000000000000000000000000000000000000 
        OR "to" = 0x0000000000000000000000000000000000000000)
    )
    SELECT 
        block_time,
        block_number,
        tx_hash,
        supply_change,
        SUM(supply_change) OVER (ORDER BY block_number, tx_hash) AS total_supply
    FROM supply_changes
    ORDER BY block_number, tx_hash
    """
    df = dune_query(sql)
    df['time'] = pd.to_datetime(df['block_time']).dt.tz_localize(None)
    return df


def save_kuru_vault_holdings(date=None, market='MONUSDC', cached_total_supply_df=None):
    if date is None:
        # run for previous date
        date = (pd.Timestamp.now() - pd.Timedelta(days=1)).floor('1D')
    
    POSTGRES_DB_URL = os.getenv('POSTGRES_DB_URL')
    conn = psycopg.connect(POSTGRES_DB_URL)
    cursor = conn.cursor()
    # if table doesn't exist, create it
    cursor.execute("CREATE TABLE IF NOT EXISTS kuru_vault_holdings (date DATE, market TEXT, timestamp_ms BIGINT, fair_value DOUBLE PRECISION, base_balance DOUBLE PRECISION, quote_balance DOUBLE PRECISION, total_balance_usd DOUBLE PRECISION, total_supply DOUBLE PRECISION)")
    conn.commit()

    start_date = pd.Timestamp(date)
    end_date = pd.Timestamp(date) + pd.Timedelta(days=1)
    df = get_strategy_state(start_date, end_date, market=market, columns=['bid_fair_value', 'ask_fair_value', 'finalized_base_balance', 'finalized_quote_balance'])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    if cached_total_supply_df is None:
        # used for bulk reloading tasks so we don't have to compute this every time
        cached_total_supply_df = get_kuru_vault_token_supply(VAULT_TOKEN_ADDRESSES[market])
    
    df = pd.merge_asof(
        df, cached_total_supply_df[['time', 'total_supply']], on='time'
    ).set_index('time')
    df = df.dropna().resample('1min', closed='right', label='right').last().reset_index()
    df['date'] = df.time.dt.date
    df['market'] = market
    df['timestamp_ms'] = df.time.apply(lambda x: x.value // 1000000)
    df['fair_value'] = (df['bid_fair_value'] + df['ask_fair_value']) / 2
    df['base_balance'] = df['finalized_base_balance']
    df['quote_balance'] = df['finalized_quote_balance']
    df['total_balance_usd'] = df['base_balance'] * df['fair_value'] + df['quote_balance']
    to_save = df[['date', 'market', 'timestamp_ms', 'fair_value', 'base_balance', 'quote_balance', 'total_balance_usd', 'total_supply']].dropna()

    cursor.executemany("INSERT INTO kuru_vault_holdings (date, market, timestamp_ms, fair_value, base_balance, quote_balance, total_balance_usd, total_supply) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", to_save.values.tolist())
    conn.commit()
    cursor.close()
    conn.close()
    return to_save


def get_parameter_changes(start_date, end_date=None):
    if end_date is None:
        end_date = pd.Timestamp.now()
    start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d %H:%M:%S')
    end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d %H:%M:%S')
    table = client.query(f"SELECT * from skew_parameter_changes where time >= '{start_str}'::timestamp and time < '{end_str}'::timestamp order by time desc")
    df = table.to_pandas()
    return df

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        date = pd.Timestamp(sys.argv[1])
    else:
        date = pd.Timestamp.now().floor('1D') - pd.Timedelta(days=1)
    save_kuru_vault_holdings(date)
