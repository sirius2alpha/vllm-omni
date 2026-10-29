# 量化（Quantization）

本文档描述 vLLM-Omni 的量化架构以及面向贡献者的扩展点。关于面向用户的配置方式、
支持的硬件以及各量化方法的具体使用说明，请参阅
[量化用户指南](../../user_guide/quantization/overview.md)。

## 目标与范围

本设计有四个目标：

1. 为纯 diffusion 模型、多阶段 omni/TTS 模型以及多阶段 diffusion 模型提供统一的
   `quantization_config` 入口。
2. 在 vLLM 后端已经提供所需能力的地方，复用 vLLM 的量化注册表与
   `QuantizationConfig` 契约。
3. 允许同一条 pipeline 中的不同组件使用不同的量化配置。
4. 保留两种加载路径的区别：对 BF16/FP16 checkpoint 做在线量化（online
   quantization），与加载带有序列化 scale 的预量化 checkpoint（pre-quantized
   checkpoint）。

本文档涵盖配置解析、组件路由、模型集成、后端扩展与验证。kernel 实现与面向用户的
支持矩阵不在本文档范围内。

## 架构

量化路径分为以下几个层次：

| 层次 | 职责 | 源码 |
| --- | --- | --- |
| 公开 API | 暴露 `build_quant_config`、组件路由与后端注册 | `vllm_omni/quantization/__init__.py` |
| 工厂 | 规范化方法名、在 Omni 覆盖实现与 vLLM 注册表后端之间做选择、检测 ModelOpt 配置、对账 checkpoint 元数据 | `vllm_omni/quantization/factory.py` |
| 组件路由器 | 基于最长前缀匹配，从层前缀解析出对应的量化配置 | `vllm_omni/quantization/component_config.py` |
| 方法配置 | 实现 `QuantizationConfig`，并选择平台相关的 linear 方法 | `vllm_omni/quantization/*_config.py` |
| Diffusion 配置 | 保存权威的运行时配置，并传播 checkpoint 中的量化配置 | `vllm_omni/diffusion/data.py` |
| Worker 与模型集成 | 把解析后的配置传入 vLLM 的模型配置与模型层 | `vllm_omni/diffusion/worker/diffusion_worker.py` 与 `vllm_omni/diffusion/models/` |

运行时契约仍然是 vLLM 的 `QuantizationConfig`：

```text
user/CLI or model config
          |
          v
OmniDiffusionConfig.quantization_config
          |
          +--> build_quant_config()
          |        |
          |        +--> Omni override
          |        +--> vLLM registry backend
          |        +--> ComponentQuantizationConfig
          |
          +--> checkpoint metadata reconciliation
          |
          +--> diffusion worker / vLLM config
          |
          +--> model layers call get_quant_method(layer, prefix)
```

模型层要么收到一个量化方法实现，要么收到 `None`。返回 `None` 表示该层保持不量化。

## 配置解析

`build_quant_config()` 接受以下几种输入形式：

| 输入 | 解析方式 |
| --- | --- |
| `None` 或 `"none"` | 关闭量化。 |
| 方法名字符串，如 `"fp8"` | 先解析 Omni 覆盖实现，再回退到 vLLM 量化注册表。 |
| 带 `method` 或 `quant_method` 字段的扁平字典 | 规范化方法名并构造对应的 `QuantizationConfig`。 |
| 按组件划分的字典（per-component dictionary） | 构造 `ComponentQuantizationConfig`。 |
| 已有的 `QuantizationConfig` 对象 | 直接透传，不重新构建。 |

方法名别名的规范化不区分大小写，`-` 与 `_` 视为等价。`auto-round` 与
`auto_round` 使用 Omni 的 INC/AutoRound 适配器。ModelOpt checkpoint 的元数据会通过
其方法名、生产者（producer）元数据或 `quant_algo` 字段被检测出来，然后映射到对应的
vLLM ModelOpt 配置。

除非输入是按组件划分的字典，否则解析出的配置会原封不动地传给 pipeline 构造的每一
个可感知量化的组件。它只影响该组件的量化方法所支持的层；它不会改写任意的
`torch.nn` 模块。按组件划分的字典则通过运行时的层前缀路由，把这个全局作用域收窄。

对于 vLLM 注册表没有提供其 diffusion 行为或平台分派的方法，工厂使用 Omni 专有的
覆盖实现。当前的覆盖集合包括 INT8、BitsAndBytes、MXFP8、MXFP4、MXFP4 双尺度
（dual-scale）、INC/AutoRound 和 TorchAO。其他方法（如 FP8、GGUF、ModelOpt）在配置
兼容时通过 vLLM 注册表解析。

## 按组件路由（Per-component routing）

多阶段模型不一定希望所有模块使用同一套量化策略。组件映射（component map）可以在
不要求每条 pipeline 分别构造 vLLM 配置的情况下表达这种策略：

```python
quantization_config = {
    "transformer": {"method": "fp8"},
    "vae": None,
    "default": None,
}
```

`ComponentQuantizationConfig` 遵循以下规则：

1. 组件的键被解释为运行时的层前缀（layer prefix）。
2. 最长匹配前缀获胜。
3. `None` 表示匹配到的组件不做量化。
4. `default`（如果存在）处理没有显式匹配的前缀。
5. 匹配到的子配置委托给其自身的 `get_quant_method(layer, prefix)` 实现。

前缀匹配发生在模型的权重名映射（weight-name mapping）之后。接入新模型时必须核实
实际的运行时前缀，特别是当 `WeightsMapper` 改写了 checkpoint 权重名的时候。前缀
不匹配会把某一层静默地路由到默认配置，因此组件路由的测试应当同时覆盖正向匹配与
落空（fall-through）行为。

## 在线量化与预量化 checkpoint

同一个方法名可能对应两种不同的加载路径：

- **在线（Online）**：源 checkpoint 包含 BF16/FP16 权重；量化方法在模型加载过程中
  创建量化权重或 scale。
- **预量化（Pre-quantized）**：checkpoint 包含量化后的权重以及该方法所需的元数据；
  加载器直接消费这些序列化张量。

模式标志由方法配置自己持有，例如 `is_checkpoint_int8_serialized` 或
`is_checkpoint_mxfp8_serialized`。配置的 `get_quant_method()` 据此选择对应的
linear 方法。

对于每个 transformer 带有独立 `config.json` 的 pipeline（如 cascade 模型），
`resolve_quant_config_from_disk()` 会把当前生效的配置与每个 transformer 的
checkpoint 元数据进行对账。其不变量（invariant）如下：

- 如果当前没有生效的配置，可以使用合法的 checkpoint 元数据进行自动检测。
- 方法不匹配是错误，而不是隐式转换。
- 检测到序列化 checkpoint 标志时，会把在线配置重建为对应的离线配置。
- checkpoint 专有的 `ignored_layers` 在二者不一致时通过重建当前配置来保留。
- AutoRound MXFP8 元数据（`data_type="mx_fp"`）选择其离线加载路径。

这样可以避免模型用在线加载器去解释序列化权重，或在多 transformer 的 pipeline 中
把错误的方法静默应用到某一个 transformer 上。

## 模型与层的集成

Diffusion 模型的构造函数接收解析后的配置，并把它传给可量化的 vLLM 层。每个方法
配置负责：

- 在 `get_quant_method()` 中识别支持的层；
- 创建正确的参数形状与加载器（loader）；
- 在需要转换时，于权重加载后进行处理（process weights after loading）；
- 在运行时应用平台相关的量化算子。

模型代码必须让精度敏感或不支持的模块保持不量化。共享的 `safe_quant_config()` 辅助
函数被模型集成代码用于归一化（normalization）与调制（modulation）层——在这些层
上套用通用 FP8 配置是不安全的。通过量化工厂暴露的编码器（encoder）与主
transformer 一样会收到全局配置，但只有它们受支持的 vLLM 可量化层会受到影响。普通
的 `torch.nn` 编码器层、VAE、scheduler 和 tokenizer 不会被自动改写。组件映射可以
收窄作用域，但无法为某一层凭空添加量化支持。

权重量化与 KV cache 量化是两条独立的路径。前者通过 `quantization_config` 解析；
运行时的 attention/KV-cache 量化由 diffusion attention 配置决定，必须独立验证。

## 平台与并行边界

平台检查属于方法实现的职责，而不是公共工厂的职责。这样可以让配置解析保持平台可
移植性，同时允许 CUDA、NPU、XPU 的方法在模型构造时选择不同的 kernel，或拒绝不支
持的执行方式。当前的硬件支持矩阵以用户指南为准。

量化方法还必须尊重所配置的并行策略产生的权重布局：

- 张量并行（TP）层可能收到被切分后的权重形状与加载器。
- 序列并行（SP）执行改变激活的搬运方式，但本身不改变量化配置。
- 数据并行（DP）的每个副本各自构建自己的方法状态；量化元数据必须在每个 rank 上
  一致可用。
- 在匹配忽略层（ignored-layer）列表或组件前缀之前，必须先应用 `WeightsMapper`
  的变换。

分布式逐层 offload（DLO）还有一条额外的边界：直接对 checkpoint 做 mmap 无法准备
在线量化的张量，因为它绕过了负责创建运行时权重与 scale 的普通加载器。逐张量
（per-tensor）的在线 FP8 linear 仍然可以通过普通加载器使用 DLO AllGather：每个
rank 先完成 FP8 张量的构建，再由 DLO 对这些运行时张量做分片，供预取
（prefetch）时重建。其他在线方法在其打包与 scale 布局完成验证之前，仍仅限于
`--dlo-no-use-allgather`。完整的兼容性矩阵见
[DLO 设计文档](offloader/distributed_layerwise_offload.md)。

## 新增一个量化后端

贡献者应按以下顺序进行：

1. 先检查现有 vLLM 后端是否能提供所需的权重格式与 kernel 行为。优先复用，而不是
   重复实现一个 Omni 后端。
2. 如果确实需要 Omni 专有行为，则在 `vllm_omni/quantization/` 中实现一个
   `QuantizationConfig` 及其 `QuantizeMethodBase`/linear 方法。
3. 实现 `get_name()`、支持的激活 dtype、最低硬件能力（minimum capability）、
   checkpoint 解析、忽略层处理以及 `get_quant_method()`。
4. 在 `factory.py` 中添加惰性（lazy）工厂覆盖。把可选的平台依赖挡在包的导入路径
   之外，并把别名统一规范化到一个权威方法名。
5. 如果该方法是预量化的，先定义 checkpoint 元数据与序列化加载路径，再添加在线
   路径。
6. 只有当模型包含必须使用不同策略的阶段时，才添加模型专有的组件路由。
7. 面向用户的可见配置写在量化用户指南中，实现层面的设计理由留在本文档。

公开的工厂注册钩子 `register_quantization_override(method, builder)` 面向这样的
集成场景：需要注册一个与 Omni 兼容的构建器（builder），但不改动 vLLM 注册表。

## 验证

验证应当覆盖完整路径，而不仅仅是配置对象能否被实例化：

| 领域 | 必查项 |
| --- | --- |
| 工厂 | 字符串、别名、扁平字典、非法输入、透传、组件映射 |
| 方法配置 | 在线/离线选择、忽略层、平台守卫、权重加载 |
| Checkpoint 元数据 | 自动检测、方法不匹配报错、序列化标志、逐 transformer 的元数据 |
| 模型集成 | 量化作用域正确、未量化组件正确、并行权重布局正确 |
| 数值质量 | 相同种子下 BF16 与量化输出在代表性分辨率和步数下的对比 |
| 性能 | 峰值显存、生成时延，以及后端特有的吞吐量指标 |

单元测试位于 `tests/diffusion/quantization/`。轨迹相似度工具
`vllm_omni/quantization/tools/compare_diffusion_trajectory_similarity.py`
为 diffusion 模型提供可复现的质量与显存对比。方法支持与特性组合仍然取决于具体模
型与硬件；仅通过工厂测试并不能证明其已达到生产可用的支持水平。

## 设计不变量（Design invariants）

- 配置规范化之后，`quantization_config` 是运行时唯一的权威来源。
- checkpoint 方法不匹配时必须明确报错，而不是冒着权重损坏的风险继续。
- 没有显式量化支持路径的组件保持原始精度。
- 可选的后端依赖一律惰性导入。
- 平台相关的能力检查与 kernel 由方法实现自己负责。
- 面向用户的支持声明，除了配置级测试之外，还需要模型、硬件、数值与显存方面的
  验证。

## MXFP8 SmoothQuant 边界

NPU 原生 MXFP8 路径支持可选的逐输入通道乘数 `mul_scale`。对一个线性层，导出器必须
配对执行以下变换，才能保持原算子等价：

```text
x_smoothed = x * mul_scale
w_smoothed = w / mul_scale[None, :]
```

运行时会在动态 MXFP8 激活量化前乘上 `mul_scale`。checkpoint 中必须已经保存由
`w_smoothed` 量化得到的 MXFP8 权重；运行时代码不会反量化并重新量化权重。该乘数沿
输入维度排列，因此 RowParallelLinear 会随输入通道切分，ColumnParallelLinear 则在
各 rank 上复制。

为兼容旧 checkpoint，`mul_scale` 默认可缺省并使用全 1 向量。合并脚本检测到对应的
`layer.div.mul_scale` 标记时，还会把 msModelSlim 的 `layer.linear.weight*` 名称还原为
运行时的 `layer.weight*`。如果合并后的 checkpoint
为每个 MXFP8 线性层都提供了乘数，合并脚本会设置 `require_smooth_scale=true`，缺失时
加载报错。

## 通用 MXFP8 Flex Smooth 校准

`calibrate_mxfp8_flex_smooth` 捕获选定 Linear 类模块的输入，并委托 MindIE-SD 搜索
alpha/beta。生成的 safetensors artifact 只包含选中层的 MXFP8 张量。artifact 元数据
提供离线 `quantization_config` 和 `ignored_layers`，让其余线性层保持基础 checkpoint
精度；`merge_mxfp8_flex_smooth_state_dict` 会在校验名称和形状后，将这些张量覆盖到基础
state dict。校准 RMSE 是层级量化仿真分数，不代表模型质量结论。
