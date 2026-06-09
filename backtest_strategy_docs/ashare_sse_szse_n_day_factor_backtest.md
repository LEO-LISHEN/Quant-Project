# A 股沪深 N 日频因子调仓回测策略

## 对应函数

```text
backtest/ashare/ashare_sse_szse_n_day_factor_backtest.py
run_ashare_sse_szse_n_day_factor_backtest
```

## 输入

### 因子打分表 `factor_score`

必须包含以下字段：

```text
ts_code
trade_date
signal_score
strategy_id
```

建议同时包含 README 中约定的完整字段：

```text
ts_code
trade_date
signal_score
signal_rank
signal_pct_rank
strategy_id
```

字段含义：

```text
ts_code       股票代码，仅处理 .SH / .SZ
trade_date    信号日期，成交发生在下一交易日开盘
signal_score  因子分数
strategy_id   策略或因子版本标识
```

### 回测底表 `backtest_base`

必须包含以下字段：

```text
ts_code
trade_date
open
close
can_buy_open
can_sell_open
is_suspended
```

建议使用沪深回测底表：

```text
base_layer/a_share/ashare_sse_szse_daily_backtest_bundle_YYYYMMDD_YYYYMMDD/
  ashare_sse_szse_daily_backtest_base.parquet
```

### 函数参数

```text
strategy_id
start_date
end_date
initial_cash
rebalance_interval
max_holdings
candidate_pool_size
weight_method
custom_rank_weights
lot_size
buy_cost_rate
sell_cost_rate
score_ascending
force_liquidate_on_last_day
annual_trading_days
plot_path
```

参数含义：

```text
strategy_id                 指定回测某个 strategy_id；为空时要求输入表只包含一个 strategy_id
start_date                  回测开始日期，格式 YYYYMMDD
end_date                    回测结束日期，格式 YYYYMMDD
initial_cash                初始资金
rebalance_interval          每 N 个交易日调仓一次
max_holdings                目标最大持仓股票数
candidate_pool_size         候选池大小，默认 max_holdings * 2，必须 >= max_holdings
weight_method               权重方式，支持 equal / custom_rank_weights
custom_rank_weights         自定义排名权重列表，长度必须等于 max_holdings，权重和 <= 1
lot_size                    买入手数单位，A 股默认 100
buy_cost_rate               买入成本率
sell_cost_rate              卖出成本率
score_ascending             是否分数越低越好；默认 False，表示分数越高越好
force_liquidate_on_last_day 是否在最后一个交易日尝试清仓
annual_trading_days         年化交易日数量，默认 252
plot_path                   可选净值曲线输出路径；.png/.jpg/.svg/.pdf 生成图片，.html 生成无依赖网页曲线
```

## 策略逻辑

T+1根据T的因子打分表，从前到后，按照候选池与目标池的上限生成候选池与目标池名单
根据T+1时的持仓情况以及T+1的开盘价确定当前持仓市值与可用资金
根据传入的权重列表确定T+1的目标池各个股票的目标值
根据该目标值准备进行调仓
调仓时对于当前持仓已不在目标池当中的股票全部卖出
仍在目标池当中，但当前市值已经超出目标值的卖出到最接近但小于等于其目标值的市值
如果存在无法卖出的持仓就不做操作
再次计算目前的当前持仓市值与可用资金
接下来用剩余可用资金按照在目标池当中的排序买入
对于目标池中未持仓的股票按最接近但小于等于其目标值买入
对于目标池中已有持仓但当前市值小于目标值的按最接近但小于等于其目标值进行补买
买入时遵循买入值最接近但小于等于其目标市值

目标池构建阶段会对明显无法买入或目标金额不足一手的新股票进行候选池递补。
目标池确定后，实际执行买入时如果因为剩余现金不足、交易成本约束或买入金额不足一手导致无法买入，则不再继续向候选池递补，剩余现金保留。





## 输出

函数返回一个字典：

```text
nav
positions
trades
rebalance_logs
metrics
config
```

### `nav`

每日组合净值表，主要字段：

```text
trade_date
cash
stock_value
total_equity
nav
num_holdings
target_num_holdings
cash_ratio
turnover
daily_return
drawdown
```

### `positions`

每日持仓明细表，主要字段：

```text
trade_date
ts_code
shares
close
last_price
market_value
actual_weight
target_weight
weight_diff
last_buy_date
is_target
is_blocked_position
```

### `trades`

成交明细表，主要字段：

```text
trade_date
signal_date
ts_code
side
price
shares
amount
cost
cash_after
reason
```

### `rebalance_logs`

调仓过程日志，包含目标池选择、成交失败、跳过原因等，主要字段：

```text
trade_date
signal_date
ts_code
action
status
reason
signal_score
target_weight
current_value
target_value
cash
```

常见 `reason`：

```text
buyable
held
cannot_buy_open
cannot_sell_open
suspended
missing_open
missing_trade_row
t_plus_one_blocked
target_value_below_one_lot
insufficient_cash_or_below_lot_size
sell_not_in_target
sell_overweight
buy_underweight_or_new_target
force_liquidate_on_last_day
```

### `metrics`

绩效指标字典，主要字段：

```text
start_date
end_date
initial_cash
final_equity
total_return
annual_return
annual_volatility
sharpe
max_drawdown
win_rate
average_turnover
total_trade_amount
total_cost
num_trades
num_buy_trades
num_sell_trades
average_num_holdings
average_cash_ratio
average_abs_weight_deviation
```

### `config`

本次回测使用的配置参数，包含实际使用的：

```text
candidate_pool_size
rank_weights
```
