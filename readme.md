# CiteVerify

引文验证工具：从 PDF/ Markdown 提取参考文献与文内引用，并校验参考文献真伪及引文-参考文献相关性。

## 功能概览

- **参考文献提取**：从 PDF URL 或本地 Markdown 提取参考文献列表（支持 MinerU API 或本地 magic_pdf）
- **引文提取**：从正文中识别并提取引用标记与对应参考文献
- **参考文献校验**：批量校验参考文献是否真实存在（多数据源检索）
- **相关性分析**：基于 LLM 分析引文与参考文献的语义相关性
- **完整流水线**：`run_full_pipeline` 一键完成 PDF → 提取 → 校验 → 相关性报告

## 环境要求

- Python 3.10+
- 使用 MinerU 本地模式时需 Python 3.10+ 及相应模型/依赖（推荐使用 MinerU API 以简化环境）

## 安装

```bash
# 克隆或进入项目目录
cd CiteVerify

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 安装依赖（根据 .venv/Lib/site-packages 导出的完整列表）
pip install -r requirements.txt
```

若仅使用 **MinerU API 模式** 与 **Web UI**，可只安装核心依赖：

```bash
pip install requests requests-toolbelt openai flask pytest
```

## 使用

### 1. 作为 Python 包

将 `src` 加入 `PYTHONPATH` 或以可安装包方式使用：

```python
import sys
sys.path.insert(0, "path/to/CiteVerify/src")

from citeverify import (
    extract_references_from_url,
    extract_references_from_markdown,
    verify_references,
    run_full_pipeline,
)
from citeverify.models import MinerUConfig, LLMConfig

# 从 PDF URL 提取参考文献
result = extract_references_from_url(
    "https://example.com/paper.pdf",
    citation_format="apa",
    listing_style="numbered",
    mineru_config=MinerUConfig(api_url="http://localhost:8000"),
)

# 完整流水线（提取 + 校验 + 相关性分析）
report = run_full_pipeline(
    pdf_url="https://arxiv.org/pdf/xxx.pdf",
    mineru_config=MinerUConfig(api_url="http://localhost:8000"),
    llm_config=LLMConfig(api_key="your-openai-key"),
)
```

### 2. Web UI

```bash
cd web-ui
python app.py
```

浏览器访问：<http://localhost:5000>。

### 3. MinerU API 服务（推荐用于 PDF 解析）

使用 Docker 启动 MinerU API，再在代码中配置 `MinerUConfig(api_url="http://localhost:8000")`：

```bash
docker run --rm -p 8000:8000 erikvullings/mineru-api:gpu
```

## 项目结构

```
CiteVerify/
├── src/
│   └── citeverify/           # 核心 Python 包
│       ├── config/           # 配置管理
│       ├── models/           # 数据模型
│       ├── utils/            # 工具函数
│       ├── llm/              # LLM 集成（OpenAI 接口）
│       ├── checkers/         # 校验与分析（文献校验、引文-参考文献相关性）
│       ├── extractor/        # 文献与引文提取
│       ├── converter/        # PDF 转 Markdown/纯文本
│       ├── pipeline.py       # 提取流水线
│       └── full_pipeline.py  # 完整验证流水线
├── web-ui/                   # Flask 前端
├── backend/                  # FastAPI 后端（若存在）
├── tests/                    # 测试
├── scripts/                  # 脚本
├── requirements.txt          # 依赖（根据 .venv/Lib/site-packages 导出）
└── readme.md
```

## 依赖说明

- `requirements.txt` 由当前 `.venv/Lib/site-packages` 导出，包含完整环境（含 MinerU 本地模式所需的 magic_pdf、torch 等）。
- 仅使用 API 模式与 Web 时，只需：`requests`、`requests-toolbelt`、`openai`、`flask`、`pytest`。

## 运行测试

```bash
# 在项目根目录，确保 src 在 Python 路径中
pytest tests/ -v
```
