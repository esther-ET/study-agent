# Research Agent

基于 LangGraph + Semantic Scholar API 的学术论文搜索智能体。

## 功能特性

- 语义模糊匹配：输入中文或英文关键词，自动解析搜索意图
- 年份限制：默认近5年，支持动态指定
- 排序规则：按时间和期刊分量排序
- Markdown 输出：标题、年份、摘要（中英对照）、链接
- 导出功能：支持 BibTeX 和 CSV 格式
- 全链路追踪：LangSmith 可视化 Agent 推理过程

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

### 运行

```bash
python -m src.main
```

### 使用示例

```
你: 找自动驾驶点云目标检测domain adaptation相关论文

搜索结果 (共找到 10 篇相关论文)

### 1. Multi-Scale Point Cloud Segmentation for Autonomous Driving
**年份**: 2023 | **来源**: CVPR | **引用数**: 150

---

### 操作选项
1. 输入论文编号选择论文
2. 输入 `export bibtex` 或 `export csv` 导出选中论文
3. 输入 `quit` 退出
```

## 技术架构

```
用户输入 → IntentParser → SearchExecutor → Formatter → Markdown 输出
                ↓
          LangSmith Tracing
```

## 项目结构

```
study_agent/
├── src/
│   ├── agent/          # LangGraph 状态机
│   │   ├── state.py    # 状态定义
│   │   ├── nodes/      # 节点实现
│   │   └── graph.py    # 图构建
│   ├── tools/          # API 和格式化工具
│   │   ├── semantic_scholar.py  # API 调用
│   │   └── formatter.py         # Markdown 格式化
│   └── utils/          # 工具函数
│       ├── tracing.py   # LangSmith 配置
│       └── export.py    # 导出功能
├── tests/              # 测试
└── configs/            # 配置
```


## 功能

- 语义模糊匹配：输入中文或英文关键词，自动解析搜索意图
- 年份限制：默认近5年，支持动态指定
- 排序规则：按时间和期刊分量排序
- Markdown 输出：标题、年份、摘要、链接
- 论文搜索：OpenAlex + arXiv 多源搜索
- 选择论文：`select 1,3` 选择多篇论文
- 导出功能：BibTeX 和 CSV 格式
- 摘要翻译：中英互译（基于 MiniMax M2.7 或 DeepSeek）
- 摘要生成：LLM 生成论文摘要（中文）
- 全链路追踪：LangSmith 可视化 Agent 推理过程

## LLM 模型配置

默认使用 MiniMax M2.7，切换方式：

```bash
# 修改 .env 文件
LLM_PROVIDER=minimax   # 或 deepseek
```

支持的模型：
- MiniMax: `minimax-m2.7`
- DeepSeek: `deepseek-chat`

## License

MIT
