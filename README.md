# Image Reidentifier

图片二次识别插件。当用户提及图片细节时，LLM 主动调用此工具对历史图片进行针对性二次识别，通过 VLM 生成更精准的结构化描述并直接返回给 LLM 上下文。

## 核心场景

用户发送图片后，Bot 首次识别生成基础描述。当后续对话中用户追问图片细节（如"左上角那个人在做什么""背景里有什么文字"），LLM 调用 `secondary_recognition` 工具，传入关注区域重新观察图片，返回结构化识别结果。

## 目录结构

```text
plugins/image_reidentifier/
├── __init__.py
├── config.py                          # 插件配置类
├── manifest.json
├── plugin.py                          # 插件入口（注册模板）
├── prompts/
│   └── secondary_recognition.yaml     # VLM 提示词模板
└── tools/
    ├── __init__.py
    └── secondary_recognition.py       # 二次识别工具
```

## 工具说明

### secondary_recognition

对已识别图片进行二次 VLM 识别，返回结构化 JSON 结果。

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `focus_area` | str | 是 | 关注区域描述（如"左上角"、"人物面部"、"背景"） |
| `image_hash` | str | 否 | 图片哈希值，精确匹配（优先级最高） |
| `image_index` | int | 否 | 图片索引，从 1 开始按时间倒序（1=最新） |
| `image_description` | str | 否 | 描述关键词，模糊匹配（优先级最低） |
| `scope` | str | 否 | 查询范围：`current_stream`（默认）/ `all_streams` |

**混合定位优先级：** `image_hash` > `image_index` > `image_description`

一旦高优先级方式定位成功，不再尝试低优先级方式。

**返回（成功）：**

```json
{
  "image_hash": "abc123...",
  "focus_area": "左上角",
  "description": "左上角有一个穿着红色衣服的女孩...",
  "objects": ["女孩", "红色衣服", "树木"],
  "confidence": 0.92,
  "model": "gpt-4o"
}
```

**返回（失败）：** `(False, "错误信息")`

## 无状态设计

本工具为纯无状态工具，不存储任何识别结果到数据库。每次调用都直接通过 VLM 识别并返回结果，识别结果由 LLM 上下文管理，无需持久化。

图片文件直接从 `Images.path` 读取原始文件，不复制。

## 执行流程

1. **混合定位** — 按优先级尝试定位图片（hash → index → description）
2. **读取图片** — 从 `Images.path` 读取文件并转 base64
3. **VLM 识别** — 使用专用提示词模板调用 VLM，注入 `focus_area`
4. **解析结果** — 从 VLM 返回中提取 JSON（支持 markdown 代码块和裸 JSON）
5. **返回结果** — 将结构化结果直接返回给 LLM

## 配置项

配置文件：`config/plugins/image_reidentifier/config.toml`

```toml
[vlm]
model = ""                      # VLM 模型名称，留空使用默认
prompt_template = "..."         # VLM 识别提示词模板
```

## 依赖

- `src.core.components.BaseTool` — 工具基类
- `src.core.models.sql_alchemy.Images` / `ImageDescriptions` / `Messages` — 数据模型
- `src.core.prompt.PromptManager` — 提示词模板管理
- `src.app.plugin_system.api.llm_api` — LLM 请求创建与模型获取
- `src.kernel.db.QueryBuilder` — 数据库查询
- `src.kernel.llm` — LLM 上下文管理与消息构建

## 版本

- Plugin: `1.0.0`
- Manifest: `plugins/image_reidentifier/manifest.json`
- Min Core: `1.2.0-rc.2`

## 📄 开源协议

本项目采用 [AGPL-v3.0](LICENSE) 协议。