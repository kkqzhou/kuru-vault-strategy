import numpy as np
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


def print_hl_hedge_report(hl_markouts, pnl_df, strategy_state, hedge_ratio=1.0):
    """
    Prints a hedge analysis summary for HL fills.

    Parameters
    ----------
    hl_markouts : pd.DataFrame
        Output of compute_markouts() for HL fills. Must have columns:
        time, is_bid, price, size, size_usd, mo_fair_value_5s
    pnl_df : pd.DataFrame
        Output of compute_trading_pnl() for HL fills.
    strategy_state : pd.DataFrame
        Strategy state with DatetimeIndex and columns: base_balance, quote_balance, fair_value, tvl, pos
    hedge_ratio : float, optional
        Fraction of vault position being hedged (e.g. 0.05). If provided, prints hedge effectiveness.
    """
    ss = strategy_state.dropna(subset=['base_balance', 'quote_balance', 'fair_value', 'tvl'])

    w = 60
    print()
    print('═' * w)
    print('  HL HEDGE ANALYSIS')
    print('═' * w)

    # --- position breakdown ---
    buys  = hl_markouts[hl_markouts['is_bid'] == True]
    sells = hl_markouts[hl_markouts['is_bid'] == False]
    buy_vol  = buys['size'].sum()
    sell_vol = sells['size'].sum()
    net_pos  = buy_vol - sell_vol          # positive = net long, negative = net short
    last_price = hl_markouts['price'].iloc[-1] if not hl_markouts.empty else float('nan')
    net_pos_usd = net_pos * last_price

    print(f"  {'POSITION':─<{w-2}}")
    print(f"  Buy  volume:   {buy_vol:>14,.2f} tokens  ({buys['size_usd'].sum():>12,.2f} USD)  [{len(buys)} fills]")
    print(f"  Sell volume:   {sell_vol:>14,.2f} tokens  ({sells['size_usd'].sum():>12,.2f} USD)  [{len(sells)} fills]")
    direction = 'LONG' if net_pos > 0 else 'SHORT'
    print(f"  Net position:  {net_pos:>+14,.2f} tokens  ({net_pos_usd:>+12,.2f} USD)  [{direction}]")
    print()

    # --- PnL breakdown ---
    final = pnl_df.iloc[-1]
    realized    = final['realized_pnl']
    unrealized  = final['unrealized_pnl']
    fees        = final['cumulative_fees']
    total       = final['total_pnl']

    print(f"  {'PnL BREAKDOWN':─<{w-2}}")
    print(f"  Realized PnL:  ${realized:>+12,.4f}")
    print(f"  Unrealized PnL:${unrealized:>+12,.4f}  (open position marked to last trade price)")
    print(f"  Fees paid:     ${-fees:>+12,.4f}")
    print(f"  ── Net total:  ${total:>+12,.4f}")
    print()

    # --- hedge effectiveness vs vault directional ---
    if not ss.empty:
        P0 = ss['fair_value'].iloc[0]
        P1 = ss['fair_value'].iloc[-1]
        b0 = ss['base_balance'].iloc[0]
        tvl0 = ss['tvl'].iloc[0]
        vault_m2m_pnl = ss['tvl'].iloc[-1] - tvl0

        # The hedge shorts the cumulative base the vault has traded away from its initial
        # inventory. At each moment it holds -hedge_ratio · (b_t - b_0) base tokens.
        # Its PnL over the period:
        #   expected_hedge_pnl = ∫ -hedge_ratio · (b_t - b_0) · dP
        # Identity: at 100% coverage, this equals HODL + Spread − Vault (i.e. the
        # rebalancing loss shown in the vault report, with opposite display sign).
        b_traded = ss['base_balance'] - b0
        dP = ss['fair_value'].diff().shift(-1)
        expected_hedge_pnl_full = -(b_traded * dP).sum()          # at 100% coverage
        expected_hedge_pnl      = expected_hedge_pnl_full * hedge_ratio
        avg_abs_traded_usd      = (b_traded.abs() * ss['fair_value']).mean()

        print(f"  {'HEDGE EFFECTIVENESS':─<{w-2}}")
        print(f"  Vault m2m PnL:                   ${vault_m2m_pnl:>+12,.2f}")
        print(f"  Avg |cumulative traded| ($):     ${avg_abs_traded_usd:>12,.2f}  (time-avg dollar exposure to hedge)")
        print(f"  Expected hedge PnL (full):       ${expected_hedge_pnl_full:>+12,.2f}  (= −∫(b_t−b₀)·dP, equals vault Rebalancing Loss)")
        ratio_label = f"{hedge_ratio*100:.1f}% coverage" if hedge_ratio != 1.0 else "full coverage"
        print(f"  Expected hedge PnL ({ratio_label:>14}): ${expected_hedge_pnl:>+12,.2f}")
        effectiveness = (total / expected_hedge_pnl * 100) if abs(expected_hedge_pnl) > 1e-6 else float('nan')
        print(f"  Actual hedge PnL:                ${total:>+12,.2f}")
        print(f"  Effectiveness:                   {effectiveness:>+11.1f}%  (100% = perfect hedge)")
        print(f"  Price move:  {P0:.5f} → {P1:.5f}  ({(P1/P0-1)*100:+.2f}%)")
        print(f"  Avg vault |pos|:  {ss['pos'].abs().mean():.4f}  (directional imbalance as fraction of TVL)")
    print('═' * w)
    print()


def print_vault_performance_report(strategy_state, markouts_df):
    """
    Prints a vault performance summary covering:
      - Mark-to-market PnL vs HODL and UniV2 benchmarks
      - PnL decomposition into spread capture vs directional
      - Inventory/position stats
      - Realised volatility of fair value

    Parameters
    ----------
    strategy_state : pd.DataFrame
        Resampled strategy state with DatetimeIndex and columns:
        base_balance, quote_balance, fair_value, tvl, pos
    markouts_df : pd.DataFrame
        Output of compute_markouts(), must have columns:
        mo_fair_value_5s, size_usd
    """
    ss = strategy_state.dropna(subset=['base_balance', 'quote_balance', 'fair_value', 'tvl'])
    if ss.empty:
        print("No strategy state data available for performance report.")
        return

    b0, q0, P0 = ss['base_balance'].iloc[0], ss['quote_balance'].iloc[0], ss['fair_value'].iloc[0]
    b1, q1, P1 = ss['base_balance'].iloc[-1], ss['quote_balance'].iloc[-1], ss['fair_value'].iloc[-1]
    tvl0 = b0 * P0 + q0

    # --- benchmarks ---
    vault_tvl   = b1 * P1 + q1
    hodl_value  = b0 * P1 + q0
    univ2_value = tvl0 * np.sqrt(P1 / P0)

    # --- PnL decomposition ---
    # Vault PnL = HODL PnL + Spread PnL - Rebalancing Loss
    # - HODL PnL:   pure directional exposure on initial inventory (the hedge should cancel this)
    # - Spread PnL: MM spread capture from markouts
    # - Rebalancing Loss: residual cost of the $x=$y constraint
    hodl_pnl         = hodl_value - tvl0
    spread_pnl       = (markouts_df['mo_fair_value_5s'] / 10000 * markouts_df['size_usd']).sum()
    vault_pnl        = vault_tvl - tvl0
    rebalancing_loss = hodl_pnl + spread_pnl - vault_pnl

    # --- realised vol (annualised, then convert to period) ---
    log_rets = np.log(ss['fair_value'] / ss['fair_value'].shift(1)).dropna()
    dt_seconds = (ss.index[1] - ss.index[0]).total_seconds() if len(ss) > 1 else 1
    periods_per_year = 365.25 * 24 * 3600 / dt_seconds
    realised_vol_ann = log_rets.std() * np.sqrt(periods_per_year)
    price_move_pct = (P1 / P0 - 1) * 100

    # --- inventory stats ---
    abs_pos = ss['pos'].abs()
    time_over_50pct = (abs_pos > 0.5).mean() * 100

    w = 60
    print()
    print('═' * w)
    print('  VAULT PERFORMANCE REPORT')
    print('═' * w)
    print(f"  Period:   {ss.index[0]}  →  {ss.index[-1]}")
    print(f"  Price:    {P0:.4f} → {P1:.4f}  ({price_move_pct:+.2f}%)   realised vol: {realised_vol_ann*100:.1f}% ann.")
    print(f"  Init TVL: ${tvl0:>12,.2f}")
    print()

    print(f"  {'PnL DECOMPOSITION':─<{w-2}}")
    print(f"  HODL PnL (initial inventory):${hodl_pnl:>+12,.4f}   ({hodl_pnl/tvl0*100:+.4f}% — exposure of b0 tokens)")
    print(f"  Spread capture (5s markout): ${spread_pnl:>+12,.4f}   ({spread_pnl/tvl0*100:+.4f}% of TVL)")
    print(f"  Rebalancing loss:            ${-rebalancing_loss:>+12,.4f}   ({-rebalancing_loss/tvl0*100:+.4f}% — cost of $x=$y)")
    print(f"  ── Total vault PnL (m2m):    ${vault_pnl:>+12,.4f}   ({vault_pnl/tvl0*100:+.4f}% of TVL)")
    print(f"  Note: hedge targets the non-rebalanced exposure (pos·tvl/2), not HODL.")
    print()

    print(f"  {'BENCHMARKS':─<{w-2}}")
    print(f"  Vault (m2m): ${vault_tvl:>12,.4f}   ({(vault_tvl/tvl0-1)*100:+.4f}%)")
    print(f"  HODL:        ${hodl_value:>12,.4f}   ({(hodl_value/tvl0-1)*100:+.4f}%)")
    print(f"  UniV2:       ${univ2_value:>12,.4f}   ({(univ2_value/tvl0-1)*100:+.4f}%)")
    print(f"  Vault vs HODL:   ${vault_tvl - hodl_value:>+12,.4f}   (pos = vault beat buy-and-hold)")
    print(f"  Vault vs UniV2:  ${vault_tvl - univ2_value:>+12,.4f}   (pos = active MM beat passive LP)")
    print(f"  Implied IL:      ${univ2_value - hodl_value:>+12,.4f}   (UniV2 - HODL, always ≤ 0)")
    print()

    print(f"  {'INVENTORY':─<{w-2}}")
    print(f"  Avg |pos|:         {abs_pos.mean():.4f}   (0=neutral, 1=fully one-sided)")
    print(f"  Max |pos|:         {abs_pos.max():.4f}")
    print(f"  Time |pos| > 50%:  {time_over_50pct:.1f}%")
    print('═' * w)
    print()


if __name__ == '__main__':
    # test trading pnl computation
    price_col = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    size_col = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    is_bid_col = pd.Series([True, True, False, False, True, False, True, True, True, True, False])
    fees_col = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11])
    pnl_df = compute_trading_pnl(price_col, size_col, is_bid_col, fees_col)
    print(pnl_df)

