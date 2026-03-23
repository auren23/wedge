# 配置管理

## 概览

Wedge 采用分层配置，优先级从高到低如下：

1. CLI 参数
2. `WEDGE_` 前缀的环境变量
3. `~/.config/wedge/config.toml` 配置文件
4. 内置默认值

旧版本遗留 key 会被忽略，因此历史配置文件不会破坏当前的 ladder-only runtime。

## 快速开始

### 初始化配置

```bash
wedge config init
```

### 设置市场凭证

```bash
wedge config set polymarket_private_key "0x..."
wedge config set polymarket_api_key "your-api-key"
wedge config set polymarket_api_secret "your-secret"
```

### 调整 Ladder 参数

```bash
wedge config set bankroll 5000
wedge config set max_bet 200
wedge config set kelly_fraction 0.2
wedge config set ladder_edge 0.08
wedge config set ladder_alloc 0.90
```

### 查看配置

```bash
wedge config show
```

## 主要命令

```bash
wedge run
wedge run --live
wedge run --bankroll 10000 --max-bet 500 --ladder-edge 0.10
wedge scan --city NYC
wedge stats --days 30
wedge backtest --days 30
```

## 配置项

### 交易参数

| Key | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `mode` | string | `"dry_run"` | `dry_run` 或 `live` |
| `bankroll` | float | `1000.0` | 初始资金 |
| `max_bet` | float | `100.0` | 单笔最大下注 |
| `kelly_fraction` | float | `0.15` | fractional Kelly 系数 |
| `max_bet_pct` | float | `0.05` | 单笔资金占比上限 |

### Ladder 策略参数

| Key | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `ladder_edge` | float | `0.08` | Ladder 最低 edge 阈值 |
| `ladder_alloc` | float | `0.90` | 预留给 ladder 的资金占比 |
| `market_min_volume` | float | `2000.0` | 市场扫描最低 24h 成交量，低于该值直接过滤 |
| `slippage_bet_size` | float | `50.0` | EV 滑点估算使用的参考下单金额 |
| `spread_baseline_f` | float | `3.0` | ensemble spread 折扣基线 |

### 退出 / 风控参数

| Key | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `fee_rate` | float | `0.02` | 盈利部分手续费 |
| `exit_loss_factor` | float | `0.5` | 相对入场价的止损触发因子 |
| `exit_min_ev` | float | `0.0` | edge 消失后的退出阈值 |
| `exit_min_hours_to_settle` | int | `12` | 距离结算太近时不提前退出 |
| `brier_threshold` | float | `0.25` | 模型质量恶化时暂停交易 |

### 调度 / 市场

| Key | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
|| `offsets_utc` | list[str] | `03:00, 09:00, 15:00, 21:00` | 调度窗口 |
| `cities` | list | 内置默认值 | 与机场对齐的城市列表 |
| `polymarket_private_key` | string | `""` | Ethereum 私钥 |
| `polymarket_api_key` | string | `""` | Polymarket API key |
| `polymarket_api_secret` | string | `""` | Polymarket API secret |

## 说明

- 预测源：direct NOAA GEFS
- 结算源：Weather Company API (Wunderground)
- 验证层：aviationweather.gov METAR
- 旧配置中的遗留 key 会被自动忽略
