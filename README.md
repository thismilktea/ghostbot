# GhostBot

GhostBot 是一个受 Claude Code 启发、基于 nanobot 框架演进而来的本地 Code Agent。它面向真实代码仓库中的长期编程协作，目标是在较低资源占用下，为开发者提供更确定、更可控的 AI 编程工作流。

与更偏通用化的代码代理不同，GhostBot 的设计重点不是“把更多上下文一次性塞给模型”，而是围绕一个项目持续积累结构认知、历史经验和运行反馈，并通过执行层约束降低模型越权、跑偏和幻觉风险。

> 当前项目处于 Alpha 阶段，核心能力正在快速迭代中，适合研究、个人开发和本地 coding agent 实验。

## 核心定位

GhostBot 关注四个方向：

- **项目级长期记忆**：以项目为单位组织记忆、历史和执行经验，而不是按一次聊天会话切割上下文。
- **优先级上下文桶管理**：将代码结构、近期对话、工具结果、长期记忆、计划状态等信息分层管理，减少无关内容干扰。
- **动态项目依赖图谱**：基于 AST、符号关系和活跃文件动态构建局部架构视野，避免全量代码喂入导致注意力稀释。
- **原子化 AI 操作与执行约束**：将模型建议、工具调用、写入权限和用户审批拆分到明确边界中，提升可控性和可追溯性。

## 为什么做 GhostBot

Claude Code、Cursor、Aider 等工具已经证明了 coding agent 的价值，但在本地长期开发场景中仍有一些痛点：

- 会话级上下文容易丢失项目历史，重启或换入口后需要重新解释项目背景。
- 大量工具结果和代码片段堆叠后，模型容易注意力漂移。
- 仅依赖提示词约束模型并不稳定，越权写入、危险命令和计划外操作需要执行层兜底。
- 全量仓库索引或向量检索在代码符号场景下容易出现语义漂移或召回噪声。
- Agent 迭代效果如果只靠主观体验判断，很难稳定优化。

GhostBot 因此更强调“项目状态”“上下文路由”“执行 harness”和“可量化评测”，希望把 coding agent 从聊天式助手推进到可持续迭代的本地开发系统。

## 主要能力

### 1. 项目级记忆，而不是会话级记忆

GhostBot 将项目作为核心状态单元。`/scan <path>` 会扫描并切换当前项目，后续对话、工具结果、计划状态和项目记忆都围绕该项目组织。

这使 Agent 可以在多轮开发和多次启动之间保留更稳定的项目认知：

- 项目结构与关键文件；
- 近期修改和对话历史；
- 用户对该项目的偏好；
- 已执行过的计划和工具反馈；
- 与当前活跃代码区域相关的记忆。

### 2. 轻量级长期记忆检索路由

针对代码场景中“向量检索容易符号漂移、纯关键词检索容易语义失真”的问题，GhostBot 保留并强化了轻量级动态记忆路由：

- 使用 FTS 粗排定位候选记忆；
- 结合逻辑规则和项目状态进行精排；
- 根据任务类型动态决定是否注入长期记忆；
- 避免把无关历史长期塞进上下文。

在项目 dogfooding 评测中，配合上下文桶和长尾工具结果截断，整体 Token 消耗降低 60% 以上，并减少了模型在长上下文中的注意力飘逸。

### 3. 优先级上下文桶管理

GhostBot 将上下文拆分为多个不同优先级的桶，例如：

- 系统约束与运行时策略；
- 当前项目结构；
- 活跃文件与工作区状态；
- 用户当前任务；
- 工具调用结果；
- 历史摘要与长期记忆；
- 计划模式状态。

不同桶使用不同的保留、压缩和截断策略。工具返回的大体量内容不会无条件进入模型上下文，而是按任务需要进行分页、压缩或延迟读取。

### 4. 动态项目依赖图谱

GhostBot 使用 Tree-sitter 增量解析代码 AST，提取符号、调用关系和文件依赖。系统会围绕当前活跃“锚点”计算局部依赖权重，并组装面向任务的局部拓扑地图。

图谱构建中引入：

- AST 符号提取；
- 调用链和引用关系；
- 时间衰减；
- 微型 PageRank 权重计算；
- 活跃文件和近期编辑反馈。

目标是让 Agent 看到“当前任务最相关的项目局部结构”，而不是盲目读取或压缩整个仓库。

### 5. 执行层 Harness，而不是只靠提示词自觉

GhostBot 将安全和权限控制下沉到工具执行层。模型不能仅凭自己决定是否可以写文件、运行命令或越过工作区边界。

运行时 Harness 会根据多维度状态实时鉴权：

- 当前运行模式；
- 用户审批策略；
- 工作区边界；
- 工具类型；
- 计划批准范围；
- 命令风险等级。

系统从“依赖模型自觉”转为“默认只读、受控写入”的确定性环境，减少越权操作、危险命令和计划外修改的风险。

### 6. 基于 Dogfooding 的自动化评测体系

GhostBot 使用自身独立开发“全栈商店项目”作为 dogfooding 测试基准，并基于 Pytest 搭建自动化评测体系。

评测指标包括：

- 单轮需求达成度；
- 工具调用冗余率；
- 自我纠错评分；
- Token 消耗；
- 响应延迟；
- 计划执行一致性。

这些指标用于量化 GhostBot 自身迭代效果，避免只依赖主观体验判断 Agent 是否真的变好。

## 与 Claude Code 类工具的差异

GhostBot 不是要复刻 Claude Code 的完整产品形态，而是更偏向研究和本地可控运行时。

| 维度 | Claude Code 类工具 | GhostBot |
| --- | --- | --- |
| 状态组织 | 通常以会话或当前工作区为主 | 以项目为核心状态单元 |
| 长期记忆 | 更依赖当前上下文和工具读取 | 项目级记忆、结构摘要和历史经验 |
| 上下文管理 | 由产品内部策略主导 | 显式上下文桶、压缩和检索路由 |
| 安全约束 | 通常有审批和沙箱能力 | 强调执行层 Harness 和工具可见性接管 |
| 项目理解 | 依赖搜索、读取和模型推理 | 结合 AST、依赖图谱和活跃锚点 |
| 评测方式 | 用户体验和任务效果为主 | Dogfooding + 自动化指标回归 |
| 定位 | 成熟通用 coding agent | 本地优先、可实验、可裁剪的 agent runtime |

## 安装

从源码安装：

```bash
git clone <your-repo-url>
cd ghostbot
pip install -e .
```

安装开发依赖：

```bash
pip install -e '.[dev]'
```

也可以使用 `uv`：

```bash
uv sync
```

## 初始化配置

推荐先运行：

```bash
ghostbot onboard
```

交互式向导：

```bash
ghostbot onboard --wizard
```

默认配置路径：

```text
~/.ghostbot/config.json
```

默认工作区：

```text
~/.ghostbot/workspace
```

也可以显式指定配置和工作区：

```bash
ghostbot agent --config ./config.json --workspace ./workspace
```

## Provider 配置

GhostBot 会根据 `agents.defaults.model` 和 `agents.defaults.provider` 选择模型后端。`provider` 默认为 `auto`，会按模型名、Provider 配置和可用 API key 自动匹配。

Anthropic 示例：

```json
{
  "providers": {
    "anthropic": {
      "apiKey": "sk-ant-..."
    }
  },
  "agents": {
    "defaults": {
      "model": "claude-opus-4-5",
      "provider": "anthropic"
    }
  }
}
```

OpenRouter 示例：

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-..."
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5",
      "provider": "openrouter"
    }
  }
}
```

本地 OpenAI-compatible 服务示例：

```json
{
  "providers": {
    "local": {
      "apiBase": "http://localhost:11434/v1"
    }
  },
  "agents": {
    "defaults": {
      "model": "llama3.2",
      "provider": "local"
    }
  }
}
```

## 常用命令

启动交互式 Agent：

```bash
ghostbot agent
```

扫描并切换当前项目：

```text
/scan C:/path/to/your/project
```

查看当前项目状态：

```text
/status
```

创建一个需要审批的执行计划：

```text
/plan 为这个项目新增用户登录功能
```

查看、修订、批准或取消计划：

```text
/plan-status --full
/plan-revise 补充单元测试和异常路径验证
/plan-approve all
/plan-cancel
```

查看计划历史和检查清单：

```text
/plan-history
/plan-checklist --full
```

清空当前项目对话线程：

```text
/new
```

停止当前项目正在运行的任务：

```text
/stop
```

发送单条消息：

```bash
ghostbot agent -m "总结这个仓库"
```

查看版本：

```bash
ghostbot --version
```

## 计划优先的编码流程

`/plan` 是 GhostBot 中最重要的工作流命令之一。它用于把一个复杂需求先转化为可审查、可修订、可分阶段执行的计划，而不是让模型直接开始改代码。

当任务涉及新增功能、修复 bug、重构或多文件修改时，可以主动使用：

```text
/plan <你的需求>
```

GhostBot 会优先进入更可控的执行流程：

1. 使用只读工具理解仓库和相关文件；
2. 生成结构化计划；
3. 等待用户批准；
4. 按批准范围执行文件修改和命令；
5. 在执行过程中持续检查计划边界；
6. 汇报执行结果、验证情况和遗留风险。

这个流程的目标不是让 Agent 变慢，而是降低长任务中偏离用户约束、误改文件或执行危险操作的概率。

## 项目结构

```text
ghostbot/
  agent/        Agent 主循环、工具、上下文、计划和执行策略
  bus/          消息总线
  cli/          命令行入口
  command/      Agent 内部 slash command
  config/       配置 schema 与加载逻辑
  project/      项目级状态、历史和记忆路由
  providers/    LLM Provider 适配
  session/      兼容层与历史状态管理
  templates/    Agent 与 planning prompt 模板
```

## 当前状态

GhostBot 仍在快速演进中，当前重点包括：

- 完成从会话级状态到项目级状态的迁移；
- 清理旧 nanobot 命名和历史兼容路径；
- 收敛非核心功能，聚焦本地 coding agent；
- 稳定项目扫描、长期记忆、上下文压缩和权限 harness；
- 扩展 dogfooding 自动化评测。

欢迎基于源码进行实验、修改和反馈。

## License

MIT
