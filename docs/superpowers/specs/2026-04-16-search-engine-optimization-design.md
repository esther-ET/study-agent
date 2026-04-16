# Study Agent 搜索引擎优化设计

> 日期: 2026-04-16
> 状态: 已实施

## 背景

2026-04-16 发现 study_agent 的论文搜索功能存在效果差的问题：
- 搜索结果不相关（返回农业、无人机、LLM 综述等无关论文）
- 排序不符合预期（高引用论文永远排前）

## 问题根因分析

### 1. OpenAlex 长查询相关性退化

**现象**: OpenAlex 对长查询（≥3个关键词）的相关性排序会退化。

**测试验证**:
- 搜 `"lidar object detection domain adaptation"` → 返回高度相关论文
- 搜 `"point cloud object detection adverse weather domain adaptation"` → 返回大量不相关论文（农业、无人机、LLM）

**原因**: OpenAlex 的 `default.search:` 机制对多关键词查询的权重分配不准确，长查询被噪声词主导。

### 2. 硬编码按引用数排序

**代码位置**: `unified_search.py:68`

```python
# 之前的代码
combined.sort(key=lambda p: p.citation_count if p.citation_count else 0, reverse=True)
```

**问题**: 不管用户选择什么排序方式，统一按引用数排序。高引用论文（Survey 类）永远排前，即使与查询无关。

### 3. filter 参数构建错误（次要）

**代码位置**: `openalex_client.py:74-76`

```python
# 之前的代码
filter_str = ",".join(filters) if filters else None
search_query = query if not filter_str else f"{query},{filter_str}"
params = {
    "filter": f"default.search:{search_query}" if search_query else None,
```

**问题**: 把年份过滤条件 `publication_year:>2020` 拼到查询字符串当作文本搜索，而不是作为真正的 filter 参数。

## 解决方案

### 1. 多路搜索策略 (Multi-Search)

**核心思想**: 对长查询分词为多个子查询，并行搜索后按匹配次数评分。

**分词策略**:
- 停用词过滤: and, or, in, on, under, with, for, the, a, an, of, to, from, by, using
- 生成所有连续 3-4 词的子串
- 特殊组合: 确保 `domain adaptation` 等关键概念被覆盖

**子查询生成示例**:
```
原始查询: "point cloud object detection adverse weather domain adaptation"
分词: ['point', 'cloud', 'object', 'detection', 'adverse', 'weather', 'domain', 'adaptation']
子查询:
  1. point cloud object detection adverse weather domain adaptation
  2. point cloud object
  3. cloud object detection
  4. object detection adverse
  5. detection adverse weather
  6. adverse weather domain
```

**评分机制**:
- 每篇论文被多少个子查询匹配到 = 匹配分数
- 最终排序: 按匹配分数降序，相同时按引用数降序

### 2. 修复 UnifiedSearch 排序逻辑

**支持 sort_by 参数**:
- `relevance`: 按多路搜索的匹配分数排序
- `citation_count`: 按引用数排序
- `year`: 按年份排序

### 3. 修复 OpenAlex filter 构建

**修复后**:
```python
if filter_parts:
    filter_str = f"default.search:{query}," + ",".join(filter_parts)
else:
    filter_str = f"default.search:{query}"
```

## 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `src/tools/openalex_client.py` | 实现多路搜索、修复 filter 构建 |
| `src/tools/unified_search.py` | 支持 sort_by 参数、修复排序逻辑 |
| `src/agent/nodes/search_executor.py` | 移除重复排序、直接传递 sort_by |

## 搜索效果对比

### 修复前
```
查询: point cloud object detection adverse weather domain adaptation

1. [无关] Unmanned aerial vehicles (UAVs): practical aspects... (879 citations)
2. [无关] A survey of uncertainty in deep neural networks (1081 citations)
3. [无关] The Path to Smart Farming: Innovations... (652 citations)
...
```

### 修复后
```
查询: point cloud object detection adverse weather domain adaptation

1. [相关] Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions (516 citations)
2. [相关] Fog Simulation on Real LiDAR Point Clouds for 3D Object Detection in Adverse Weather (182 citations)
3. [相关] Towards A Weakly Supervised Framework for 3D Point Cloud Object Detection (132 citations)
...
```

## 后续优化方向

1. **语义搜索增强**: 考虑使用 sentence-transformers 等 embedding 模型计算语义相似度
2. **多数据源协同**: 整合 arXiv 的 CS/ML 专精搜索能力
3. **查询扩展**: 自动识别同义词、近义词进行查询扩展
4. **结果去重**: 跨数据源去重，避免同一论文重复出现
