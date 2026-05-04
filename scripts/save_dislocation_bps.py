from influxdb_client_3 import InfluxDBClient3
import os
import dotenv
import sys
import pandas as pd
dotenv.load_dotenv()

output_dir = os.path.join(os.path.dirname(__file__), '../data')

INFLUXDB_URL = os.getenv('KURU_MD_INFLUXDB_URL')
INFLUXDB_TOKEN = os.getenv('KURU_MD_INFLUXDB_TOKEN')
DATABASE = os.getenv('KURU_MD_INFLUXDB_DATABASE')

client = InfluxDBClient3(host=INFLUXDB_URL, token=INFLUXDB_TOKEN, database=DATABASE)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python save_dislocation_bps.py <date>')
        sys.exit(1)
    date = sys.argv[1]
    venue = 'coinbase'
    start_date = pd.Timestamp(date)
    t1 = start_date + pd.Timedelta(days=0.5)
    t2 = start_date + pd.Timedelta(days=1)
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    t1_str = t1.strftime('%Y-%m-%dT%H:%M:%SZ')
    t2_str = t2.strftime('%Y-%m-%dT%H:%M:%SZ')
    sql1 = f"""
    SELECT * FROM dislocation_data where time >= '{start_date_str}' and time < '{t1_str}' and venue = '{venue}'
    """
    sql2 = f"""
    SELECT * FROM dislocation_data where time >= '{t1_str}' and time < '{t2_str}' and venue = '{venue}'
    """
    df1 = client.query(sql1).to_pandas()
    df2 = client.query(sql2).to_pandas()
    df = pd.concat([df1, df2])
    df.to_csv(os.path.join(output_dir, f'dislocation_bps_{venue}_{start_date.strftime('%Y%m%d')}.csv'), index=False)
