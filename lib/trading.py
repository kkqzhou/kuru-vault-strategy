import pandas as pd

def process_timestamp_column(series):
    series_ = pd.to_datetime(series, format='mixed').dt.tz_localize(None)
    units = ['us', 'ms', 's']
    i = -1
    while series_.max() < pd.Timestamp('1971-01-01'):
        i += 1
        series_ = pd.to_datetime(series, unit=units[i], format='mixed').dt.tz_localize(None)
    return series_
        
def format_for_timestamp_index(_df):
    if isinstance(_df.index, pd.DatetimeIndex):
        _df.index = _df.index.tz_localize(None)
        return _df

    cols = ['time', 'timestamp', 'timestamp_ms']
    if _df.index.name in cols:
        _df.index = _df.index.tz_localize(None)
        _df = _df.reset_index()

    assert len(set(cols).intersection(_df.columns)) == 1, "trades must have exactly one of the columns: time, timestamp, timestamp_ms. Got columns: " + str(list(_df.columns))
    for col in cols:
        if col in _df.columns:
            _df[col] = process_timestamp_column(_df[col])
            _df = _df.set_index(col)
            return _df
    return _df


def compute_markouts(trades, fair_values, lags=[0, 1, 5, 15, 60]):
    assert 'is_bid' in trades.columns, "trades must have a boolean column named is_bid"
    assert 'price' in trades.columns, "trades must have a float column named price"
    trades_ = trades.copy()
    trades_ = format_for_timestamp_index(trades_)
    assert isinstance(fair_values.index, pd.DatetimeIndex), "fair_values must have a timestamp index. Got:\n" + str(fair_values)
    for lag in lags:
        s = fair_values.copy()
        s.index -= pd.Timedelta(seconds=lag)
        name = fair_values.name + f'_{lag}s'
        s.name = name
        trades_ = pd.merge_asof(trades_, s, on='time')
        trades_[f'mo_{name}'] = (trades_[name] / trades_['price'] - 1) * 10000 * (2 * trades_['is_bid'] - 1)
    return trades_


def compute_trading_pnl(price_col, size_col, is_bid_col, fees_col=None, time_index=None):
    # assumes that all of these columns are time ordered and that size_col is strictly positive
    # returns unrealized_pnl, realized_pnl
    # fee must be expressed as an absolute amount and not as a rate or percentage
    if time_index is None:
        time_index = price_col.index.copy()
    if fees_col is None:
        fees_col = pd.Series(0, index=time_index)
    assert isinstance(price_col, pd.Series), "price_col must be a pandas Series"
    assert isinstance(size_col, pd.Series), "size_col must be a pandas Series"
    assert isinstance(is_bid_col, pd.Series), "is_bid_col must be a pandas Series"
    assert isinstance(fees_col, pd.Series), "fees_col must be a pandas Series"
    assert len(price_col) == len(size_col) == len(is_bid_col) == len(fees_col), f"price_col {len(price_col)}, size_col {len(size_col)}, is_bid_col {len(is_bid_col)}, and fees_col {len(fees_col)} must have the same length"

    signed_size_col = (2 * is_bid_col - 1) * size_col
    realized_pnls = []
    unrealized_pnls = []
    total_pnls = []
    cumulative_fees = []

    realized_pnl = 0
    pos = 0
    cost = 0
    running_pos = 0
    running_cost = 0
    running_fee = 0
    for price, signed_size, fee in zip(price_col, signed_size_col, fees_col):
        if (pos >= 0 and signed_size > 0) or (pos < 0 and signed_size < 0):
            pos += signed_size
            cost += price * signed_size
        elif pos >= 0 and signed_size < 0:
            cost_basis = price if pos == 0 else cost / pos
            if pos + signed_size >= 0:
                # partial close of long position
                realized_pnl += abs(signed_size) * (price - cost_basis)
                pos += signed_size
                cost = cost_basis * pos
            else:
                # entire long position is closed and residual short is opened
                realized_pnl += pos * (price - cost_basis)
                cost = (pos + signed_size) * price
                pos += signed_size
        elif pos < 0 and signed_size > 0:
            cost_basis = price if pos == 0 else cost / pos
            if pos + signed_size <= 0:
                # partial close of short position
                realized_pnl += abs(signed_size) * (cost_basis - price)
                pos += signed_size
                cost = cost_basis * pos
            else:
                # entire short position is closed and residual long is opened
                realized_pnl += abs(pos) * (cost_basis - price)
                cost = (pos + signed_size) * price
                pos += signed_size

        unrealized_pnls.append(pos * price - cost)
        realized_pnls.append(realized_pnl)

        running_fee += fee
        cumulative_fees.append(running_fee)
        running_pos += signed_size
        running_cost += price * signed_size
        total_pnls.append(running_pos * price - running_cost - running_fee)
    
    pnl_df = pd.DataFrame({'unrealized_pnl': pd.Series(unrealized_pnls), 'realized_pnl': pd.Series(realized_pnls), 'cumulative_fees': pd.Series(cumulative_fees), 'total_pnl': pd.Series(total_pnls)})
    pnl_df.index = time_index
    if (pnl_df.total_pnl - pnl_df.unrealized_pnl + pnl_df.cumulative_fees - pnl_df.realized_pnl).abs().max() > 1e-9:
        pnl_df['diff'] = pnl_df.total_pnl - pnl_df.unrealized_pnl + pnl_df.cumulative_fees - pnl_df.realized_pnl
        print(pnl_df[pnl_df['diff'].abs() > 1e-9])
        raise AssertionError("Total PnL does not match the sum of unrealized, cumulative fees, and realized PnL:\n" + str(pnl_df))
    return pnl_df


def print_trading_report(df_with_markouts, markout_col, aux_print_cols, time_index=None, price_col_name='price', size_col_name='size', is_bid_col_name='is_bid', fee_col_name='fee', print_report=True):
    assert 'size_usd' in df_with_markouts.columns, "df_with_markouts must have a column named size_usd"
    duration = markout_col.split('_')[-1]
    if markout_col not in aux_print_cols:
        aux_print_cols.append(markout_col)

    fees_col = None if fee_col_name is None else df_with_markouts[fee_col_name]
    pnl_df = compute_trading_pnl(
        df_with_markouts[price_col_name], df_with_markouts[size_col_name], df_with_markouts[is_bid_col_name], fees_col=fees_col, time_index=time_index
    )
    if print_report:
        print(f'Mean markout at {duration}:      ', df_with_markouts[markout_col].mean().round(4))
        print(f'$ weighted markout at {duration}:', ((df_with_markouts[markout_col] * df_with_markouts['size_usd']).sum() / df_with_markouts['size_usd'].sum()).round(4))
        print(f'\nWorst 10 markouts at {duration}:')
        print(df_with_markouts[aux_print_cols].sort_values(markout_col).head(10))
        print(f'\nTrading PnL (Volume = ${df_with_markouts["size_usd"].sum().round()}):')
        print(pnl_df)

    return pnl_df


if __name__ == '__main__':
    # test trading pnl computation
    price_col = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    size_col = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    is_bid_col = pd.Series([True, True, False, False, True, False, True, True, True, True, False])
    fees_col = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11])
    pnl_df = compute_trading_pnl(price_col, size_col, is_bid_col, fees_col)
    print(pnl_df)

