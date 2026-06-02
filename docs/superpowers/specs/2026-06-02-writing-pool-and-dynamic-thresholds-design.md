# Writing-Port 加权模型桶 + Flash-Upgrade 动态阈值控制器

**日期**: 2026-06-02
**状态**: 已批准，待实现
**范围**: 两个相关特性，共享同一次实现

---

## 1. 背景与目标

当前架构下，每个 port 由 `config.provider_for_port(port)` 绑定**唯一** provider，该 `Provider`
对象贯穿整个请求管道（模型名规范化、reasoning_effort 注入位置、`has_reasoning_content`、
per-provider 升格阈值、`pro_model`/`flash_model`、cross-consult、`/v1/models`）。

本次引入两个特性：

- **特性 1 — Writing-port 加权模型桶**：writing_port（8001）按加权随机从一个跨 provider 家族的
  模型池中**逐请求**选取一个 `(provider, model)`，权重当前 1:1:1:1。BERT/启发式升格在 flash 起始
  模型上仍然有效。
- **特性 2 — Per-port 动态阈值控制器**：对**所有** port 的 BERT (`router_threshold`) 与启发式
  (`heuristic_threshold`) 升格阈值增加闭环反馈调整，阈值在配置值 ±20% 带内浮动，把**升格可及请求**
  的实际升格率驱动到 `1 − flash_floor`（默认 60%，即 40% flash 均衡），保证快速模型份额不低于
  `flash_floor`。

### 已锁定的设计决策

| 决策 | 选择 |
|---|---|
| 桶选择粒度 | **逐请求重掷**（无会话粘滞，可每次工具调用重掷） |
| 阈值调整机制 | **闭环反馈控制器**（观测实际比例后调整） |
| 比例统计母体 | **仅升格可及请求**（flash 起始；pool 直接选中的 pro 不计入） |
| 控制律 | **双向驱动到 `flash_floor` 均衡点**（f 以 1.0 为中心，两侧 ±20%） |
| `flash_floor` | **可调参数**（全局，默认 0.40） |

---

## 2. 特性 1 — Writing-port 加权模型桶

### 2.1 配置 schema

`PortBinding` 新增可选字段 `model_pool`：

```yaml
ports:
  - port: 8000
    provider: deepseek
    sampling: precise
  - port: 8001
    sampling: creative
    provider: mimo                      # home provider：/v1/models 默认 + pool 解析兜底
    model_pool:
      - { provider: deepseek, model: deepseek-v4-flash, weight: 1 }
      - { provider: mimo,     model: mimo-v2.5,         weight: 1 }
      - { provider: deepseek, model: deepseek-v4-pro,   weight: 1 }
      - { provider: mimo,     model: mimo-v2.5-pro,     weight: 1 }
```

新增 pydantic 模型 `PoolEntry`（在 `providers.py`）：

```python
class PoolEntry(BaseModel):
    provider: str           # 必须匹配 providers 字典中的 key
    model: str              # 必须等于该 provider 的 flash_model 或 pro_model
    weight: float = Field(default=1.0, gt=0)   # 相对权重
```

`PortBinding` 增加：

```python
model_pool: list[PoolEntry] | None = Field(default=None,
    description="加权随机模型桶。给定时，该 port 逐请求从池中选 (provider, model)，"
                "覆盖单一 provider 路由。provider 字段仍作为 home/兜底。")
```

`provider` 字段**保持必填**（home provider，用于 `/v1/models` 默认与兜底）。

### 2.2 校验

在 `ProxyConfig` 的 `model_validator(mode="after")` 中（此时 `providers` 已就绪）：

- 每个 `PoolEntry.provider` 必须存在于 `self.providers`。
- 每个 `PoolEntry.model` 必须等于该 provider 的 `flash_model` 或 `pro_model`
  （保证升格语义良定：flash→可升格，pro→pin）。
- `model_pool` 非空、所有 `weight > 0`（`weight` 的 `gt=0` 由字段级保证；额外校验列表非空）。
- 违反任一条 → `ValueError`，启动期失败（fail-fast）。

### 2.3 选择器

新增模块 `deep_proxy/pool.py`：

```python
def select_pool_target(
    binding: PortBinding, config: ProxyConfig, *, rng=random,
) -> tuple[Provider, str]:
    """对 binding.model_pool 加权随机选一个条目，返回 (Provider, model_id)。
    rng 注入以便测试用 seeded RNG。binding.model_pool 必须非空（调用方保证）。"""
```

实现：`rng.choices(entries, weights=[e.weight for e in entries], k=1)[0]` → 解析
`config.providers[entry.provider]` 与 `entry.model`。

### 2.4 接线（main.py）

`_binding_for_request` 改为：解析 binding 后，若 `binding.model_pool` 非空，调用
`select_pool_target` 得到 `(provider, model)`，返回时附带选中的 model（供端点覆盖 `body["model"]`）。

签名调整为返回 `(provider, sampling, port, selected_model)`（`selected_model` 仅 pool 时非 None）。
两个 chat 端点（OpenAI `/v1/chat/completions` 与 Anthropic `/v1/messages`）在调
`_maybe_redirect_provider` **之前**：

```python
if selected_model is not None:
    body["model"] = selected_model
```

选择发生在 redirect 之前 —— `_maybe_redirect_provider` 需要已解析的 provider，cross-consult
标签重定向在选中的 provider 之上正确组合（如 pool 选中 mimo、用户打了"换家族"标签 → redirect 翻到
deepseek）。

`prepare_request` 其余逻辑**不变**：
- flash 起始（`body["model"] == provider.flash_model`）→ `_maybe_upgrade` 触发，可升 pro。
- pro 起始（`body["model"] == provider.pro_model`）→ `_maybe_upgrade` 不触发 → **pin 在 pro**，
  不走 hysteresis 降格（Direction C 仅在 flash 路径内）。

### 2.5 `/v1/models`

pool 配置时列出**池内 provider 家族的并集**。`router.list_models` 增加可选参数
`pool_providers: list[Provider] | None`：给定时对每个 provider 取其模型列表（deepseek 走
上游拉取+本地兜底；mimo 走本地 `MIMO_MODELS`），按 home provider 优先排序，按 `id` 去重并集。
未给定 pool_providers 时保持现有单 provider 行为。

`main.py` 的 `/v1/models` 端点：binding 有 pool → 收集池内**去重的** Provider 列表传入。

---

## 3. 特性 2 — Per-port 动态阈值控制器

### 3.1 控制器

新增 `deep_proxy/optimization/dynamic_threshold.py`：

```python
class DynamicThresholdController:
    """单 port 的闭环阈值调整器。
    - 滑动窗口（deque maxlen=window）记录**阈值驱动**的升格决策（upgraded: bool）
    - 当前因子 f（init 1.0），施加到 router_threshold 与 heuristic_threshold（同向缩放）
    """
    def __init__(self, *, flash_floor=0.40, band=0.20, window=50, kp=0.5, min_samples=10): ...

    @property
    def factor(self) -> float:
        """当前 f，clamp 在 [1-band, 1+band]。样本数 < min_samples 时返回 1.0（暖机）。"""

    def record(self, upgraded: bool) -> None:
        """记录一次阈值驱动决策并更新 f。"""
```

**控制律**（比例控制，双向驱动到均衡）：

```
target_upgrade_rate = 1 - flash_floor          # 默认 0.60
if len(window) < min_samples:
    f = 1.0                                     # 暖机
else:
    upgrade_rate = sum(window) / len(window)
    error = upgrade_rate - target_upgrade_rate  # >0：升格过多 → 抬高阈值压制
    f = clamp(1.0 + kp * error, 1 - band, 1 + band)
```

- `upgrade_rate > target`（flash < floor）→ `f > 1.0` → 阈值↑ → 升格更难 → flash 回升。
- `upgrade_rate < target`（flash > floor）→ `f < 1.0` → 阈值↓ → 升格更多 → 拉向均衡。
- 均衡在 `upgrade_rate = target`（flash = floor），此时 f≈1.0（尊重配置阈值）。
- f 饱和在 ±band 时记 `logger.info`（带 saturation 标记），不抛错。

### 3.2 阈值施加

`UpgradeDecisionEngine._resolve_params` 在算出 per-provider base 阈值后乘以 f：

```
router_thr_eff = clamp(base_router_thr * f, 0.0, 1.0)
heur_thr_eff   = clamp(base_heur_thr * f, downgrade_thr + EPS, 10.0)
```

`heur_thr_eff` 的下钳保住 hysteresis 不变式（`heuristic > downgrade`）。f 读取发生在决策**之前**
（反映过去窗口）。

### 3.3 母体正确性（关键）

控制器**只**记录 Step 3/4（启发式+BERT）的阈值驱动决策，记录点在
`_step_compute_and_commit` 末尾（拿到最终 `did_upgrade` 后 `controller.record(did_upgrade)`）。

**排除**出窗口：
- Step 1 sentinel / `extra_body` 强制升格（非阈值驱动）
- Step 2a throttle 冷却强制 flash（非阈值驱动）
- Step 2b 持久缓存命中（非阈值驱动；hysteresis 降格 fall-through 到 Step 3/4 时才记录）
- pool 直接选中的 pro：`model == pro_model` → `_maybe_upgrade` 根本不触发，天然不进引擎

⇒ 窗口母体 = 恰好"阈值能左右其结果"的请求集合，与"仅升格可及请求"语义一致。

### 3.4 接线

- `prepare_request` 新增 `port: int | None = None`。
- `_maybe_upgrade` / `engine.apply` / `_resolve_params` / `_step_compute_and_commit` 新增
  `controller: DynamicThresholdController | None = None`（**默认 None → f=1.0 → 与现状完全等价**，
  46 个 `test_flash_upgrade` 用例不改即通过）。
- `DeepProxyRouter` 持有 `self._threshold_controllers: dict[int, DynamicThresholdController]`，
  按 port 惰性创建（仅 `dynamic_threshold.enabled` 时）。`prepare_request` 按 `port` 解析控制器
  并传入 `_maybe_upgrade`。
- `main.py` 的 `_binding_for_request` 已知 port；chat 端点把 port 传入 `prepare_request`。

### 3.5 配置 schema

新增 `DynamicThresholdConfig`（嵌套于 `FlashUpgradeConfig`）：

```yaml
flash_upgrade:
  # ... 现有字段 ...
  dynamic_threshold:
    enabled: true
    flash_floor: 0.40    # 均衡点 flash 份额（可调，0 < x < 1）
    band: 0.20           # 阈值 ±调整带（可调，0 <= x <= 1）
    window: 50           # 滑动窗口样本数（可调，>= 1）
    kp: 0.5              # 比例增益（可调，> 0）
    min_samples: 10      # 暖机阈值（可调，>= 1）
```

```python
class DynamicThresholdConfig(BaseModel):
    enabled: bool = True
    flash_floor: float = Field(default=0.40, gt=0.0, lt=1.0)
    band: float = Field(default=0.20, ge=0.0, le=1.0)
    window: int = Field(default=50, ge=1)
    kp: float = Field(default=0.5, gt=0.0)
    min_samples: int = Field(default=10, ge=1)
```

全局生效（一份配置应用到每个 port，各 port 独立窗口/f 状态）。per-port `flash_floor` 覆盖
留作未来扩展（YAGNI，本次不做）。

---

## 4. 文件改动清单

| 文件 | 改动 |
|---|---|
| `providers.py` | 新增 `PoolEntry`；`PortBinding` 加 `model_pool` |
| `pool.py`（新） | `select_pool_target` |
| `config.py` | 新增 `DynamicThresholdConfig`；`FlashUpgradeConfig` 加 `dynamic_threshold`；`ProxyConfig` model_validator 校验 pool 条目 |
| `optimization/dynamic_threshold.py`（新） | `DynamicThresholdController` |
| `optimization/upgrade_decision.py` | `apply`/`_resolve_params`/`_step_compute_and_commit` 加 `controller`，施加 f + 记录决策 |
| `router.py` | `prepare_request` 加 `port`；`_maybe_upgrade` 加 `controller`；`_threshold_controllers` 注册表 |
| `main.py` | `_binding_for_request` 做 pool 选择 + 暴露 port；两个 chat 端点覆盖 `body["model"]` + 传 port；`/v1/models` 并集 |
| `models_list.py` / `router.list_models` | `pool_providers` 并集支持 |
| `config.yaml` / `config.example.yaml` | 写入 8001 pool + `dynamic_threshold` 块 |

---

## 5. 测试计划

**单元**
- `select_pool_target`：seeded-RNG 权重分布、provider/model 解析、空池/缺失 provider/非 flash/pro model 校验报错。
- `DynamicThresholdController`：f 轨迹、±band 钳制、`flash_floor` 均衡收敛、暖机（< min_samples → f=1.0）、窗口驱逐、saturation 日志。
- `ProxyConfig` pool 校验：合法/非法 config 加载。

**管道**
- writing-port 各 4 种 pick → 正确 provider 贯穿；flash 升格可及、pro pin；pool-pro 与 sentinel/throttle/persist 不进控制器窗口。
- 控制器 `enabled=false` 或 `controller=None` → 行为与现状完全等价（46 个 `test_flash_upgrade` + 现有管道用例全绿）。
- `dynamic_threshold` 施加 f 后 `_resolve_params` 阈值正确缩放且钳制。

**模型列表**
- pool 配置 → `/v1/models` 返回 deepseek+mimo 并集、去重、home 优先。
- coding port 不受影响。

---

## 6. 非目标（YAGNI）

- 会话粘滞选择（已明确逐请求重掷）。
- per-port / per-provider `flash_floor` 覆盖。
- 控制器持久化（进程内状态，重启清零可接受）。
- pool-pro 直接选中计入比例母体（已明确仅升格可及）。
