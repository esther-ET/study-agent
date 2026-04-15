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

## License

MIT
