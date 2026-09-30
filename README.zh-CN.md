# CoatWeave Harness

[English](README.md) · [工具接口](docs/TOOLS.md) · [路线图](docs/ROADMAP.md) · [离线示例](examples/README.md)

**用切片定位专利，用全文工具查证证据。**

长期目标是一个完整的、工具驱动的 coating research agent：借鉴 Codex / Claude Code 的任务工作区与工具交互方式，围绕涂料研究任务规划、调用工具、调查证据并记录过程。

**仓库包含原有 Python 检索基础层，以及独立的 Pi SDK 服务层。** 本项目不声称具备完整自主涂料研发能力，也不声称具备 Codex / Claude Code 的完整能力或官方关联。

## Pi 服务层

[pi-harness/](pi-harness/README.md) 迁入经授权并脱敏的会话 API、原生事件 SSE、兼容 completion 接口、`patent_search` 工具和 FastAPI/Milvus/Markdown 后端。使用固定版本的官方 Pi SDK，不复制整个上游仓库。

默认只读、本地回环地址、独立运行目录；写入工具可显式开启，仅允许写入工作区 `outputs/`。两种 API 使用统一的 11 字段元数据日志，不记录正文、推理、原始工具参数或返回内容。原生会话持久化文件仍可能含敏感内容，必须留在私有目录。

原生 `/sessions/{id}/events` 可以提供工具调用状态；兼容 completion 接口目前仅流式返回正文和心跳，不能宣称已经向它增加工具进度事件。共享 Bearer token 不是多租户鉴权；shell 单独开启且不受文件路径规则限制。

迁移去掉了真实主机、部署账号、固定桶名、集合审计信息与私有提示词。模型、数据库、文档和凭证需要使用自己的配置。本次没有接入超图 RAG，也没有自动把原 Python CLI 接到 Pi；两条检索实现保持独立。

```bash
cd pi-harness
npm ci --ignore-scripts
npm run check
npm test
npm run build
npm start
```

启动配置、可选写入、数据保留和安全边界见 [服务说明](pi-harness/README.md)，离线检查见 [验证记录](docs/PUBLICATION_CHECKS.md)。健康检查和固定欢迎语不能证明真实模型或数据库已接通。

## 为什么不只匹配 chunk？

配方、制备条件、性能表和对比例可能分散在同一专利的不同章节。一个向量命中的 chunk 可以定位文档，却未必包含解释结果需要的完整上下文。

本实现将发现与调查分开：

```text
问题 → chunk 精确余弦召回 → 按专利公开编号聚合
     → 标题与代表 chunk 重排 → 候选专利完整清洗 Markdown
     → Agent 多轮 find / grep / read → 带行号的证据回答
```

当前 CLI 配置先选最多 10 篇不同专利，再重排保留最多 5 篇。之后读取的是入选专利的完整清洗 Markdown，不是只把命中的几个 chunk 塞给模型。Agent 可以继续跨章节找证据；但被初筛或重排排除的专利，目前无法由它重新召回。

## 已实现与未实现

以下说明描述原有 Python CLI，不是新增 Pi 服务的接口清单。

- 已实现：MinerU 输出清洗、章节切片、精确向量检索、专利聚合、代表 chunk 重排、全文文件工具、多轮流式调用与工具轨迹。
- 提示词要求：搜索同义词和 OCR 词干、读上下文、区分实验与对比、核对单位与条件、逐结论引用行号。
- 尚未自动验证：是否充分搜索、是否成功使用全部三个工具、引用是否真实支持结论。
- 全文指清洗后文本，不是原 PDF 的完整视觉内容；复杂表格和图片理解仍有限。
- 路径检查限制文件工具访问范围，但不是操作系统级安全沙箱。

## 从离线验证开始

从仓库根目录运行，Python 3.10+：

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy PyYAML python-dotenv
python -m pip install --no-deps -e .
python -m unittest discover -s patent_coating/tests -v
coat-weave-harness --help
```

这个轻量环境仅用于离线测试、帮助和文件工具，不满足真实模型运行的全部依赖。Windows 若无创建符号链接权限，路径逃逸测试会在建立测试夹具时失败；不能把它当作测试通过。

首次公开验证：原始测试在 WSL Ubuntu / Python 3.12 下 16/16 通过；Windows 为 15 项通过、1 项因创建符号链接权限不足报错，未修改或静默跳过测试。可编辑安装与 CLI 帮助通过。这些不构成真实模型效果或完整入库验收。仓库附带 Linux 离线 CI，其运行结果独立记录。

真实运行请安装 `python -m pip install -e .`，使用自己的模型与 MinerU 输出，修改 [配置](patent_coating/patent_rag_config.yaml)，并将密钥只放入本地 `patent_coating/.env`。相对模型路径要求从仓库根目录运行；数据路径相对配置文件解析。`ask` 会向配置的提供商发送问题和候选专利文本，可能收费，运行前确认权限与数据策略。

原始专利、OCR 输出、解析全文、向量索引、模型权重和真实密钥均不随公开仓库发布。演示数据为明确标注的虚构文本。

## 接下来

Pi 服务已有独立的 Milvus 专利工具；将原 Python 检索链与超图 RAG 接入、扩展证据核验及其他涂料研究工具仍是后续工作。详见 [Roadmap](docs/ROADMAP.md)，不能将这些计划写成已交付功能。

作者：[Nathan10969](https://github.com/Nathan10969)。软件及原创插图使用 [MIT](LICENSE)；不覆盖第三方专利、模型权重或服务条款。
