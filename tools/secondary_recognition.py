"""image_reidentifier 二次识别工具。

提供 secondary_recognition 工具，供 LLM 调用以对已识别图片进行二次识别，
通过 VLM 生成更精准的结构化描述。

支持：
- focus_area 参数：针对性识别图片特定区域
- 混合定位：image_hash > image_index > image_description
- scope 参数：current_stream / all_streams
- 结构化 JSON 返回：focus_area + description + objects + confidence
"""

from __future__ import annotations

import ast
import base64
import json
from pathlib import Path
from typing import Annotated

from src.app.plugin_system.api.llm_api import create_llm_request, get_model_set_by_task
from src.core.components import BaseTool
from src.core.models.sql_alchemy import Images, Messages
from src.core.prompt import get_prompt_manager
from src.kernel.db import QueryBuilder
from src.kernel.llm import LLMContextManager, LLMPayload, ROLE, Text, Image
from src.kernel.logger import get_logger

logger = get_logger("image_reidentifier.secondary_recognition")

# 文件扩展名到 MIME 类型的映射
_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class SecondaryRecognitionTool(BaseTool):
    """图片二次识别工具。

    对已识别图片进行二次识别，通过 VLM 生成更精准的结构化描述。

    支持混合定位（image_hash > image_index > image_description）、
    focus_area 针对性识别、scope 跨流查询。
    """

    name: str = "secondary_recognition"
    description: str = (
        "对已识别图片进行二次识别，通过VLM生成更精准的描述。"
        "支持focus_area指定关注区域、混合定位（image_hash/image_index/image_description）、"
        "scope跨流查询（current_stream/all_streams）。"
    )

    async def execute(
        self,
        focus_area: Annotated[str, "关注区域描述（如：左上角、人物面部、背景等）"],
        image_hash: Annotated[str | None, "图片哈希值（精确匹配，优先级最高）"] = None,
        image_index: Annotated[int | None, "图片索引（从1开始，按时间倒序，1=最新）"] = None,
        image_description: Annotated[str | None, "描述关键词（模糊匹配，优先级最低）"] = None,
        scope: Annotated[str, "查询范围：current_stream（仅当前流）/ all_streams（全局）"] = "current_stream",
    ) -> tuple[Annotated[bool, "是否成功"], Annotated[str | dict, "返回结果"]]:
        """执行图片二次识别。

        Args:
            focus_area: 关注区域描述（如：左上角、人物面部、背景等）
            image_hash: 图片哈希值（精确匹配，优先级最高）
            image_index: 图片索引（从1开始，按时间倒序，1=最新）
            image_description: 描述关键词（模糊匹配，优先级最低）
            scope: 查询范围（current_stream 仅当前流 / all_streams 全局）

        混合定位优先级：image_hash > image_index > image_description

        Returns:
            tuple[bool, str | dict]: (是否成功, 识别结果)
                成功时返回包含 focus_area/description/objects/confidence 的字典；
                失败时返回错误信息字符串。
        """
        logger.info(
            f"二次识别请求: focus_area={focus_area}, "
            f"image_hash={image_hash}, image_index={image_index}, "
            f"image_description={image_description}, scope={scope}"
        )

        # ── 1. 混合定位：按优先级尝试定位图片 ──
        image_id = await self._locate_image(
            image_hash=image_hash,
            image_index=image_index,
            image_description=image_description,
            scope=scope,
        )
        if image_id is None:
            msg = (
                "未找到图片记录。请提供 image_hash、image_index 或 image_description 之一。"
            )
            logger.warning(msg)
            return False, msg

        # ── 2. 查询 Images 表获取图片信息 ──
        media = await (
            QueryBuilder(Images)
            .filter(image_id=image_id)
            .order_by("-timestamp")
            .first()
        )
        if media is None:
            msg = f"未找到图片记录: image_id={image_id}"
            logger.warning(msg)
            return False, msg

        # ── 3. 读取图片文件并转为 base64 ──
        image_path = Path(media.path)
        if not image_path.exists():
            msg = f"图片文件不存在: {image_path}"
            logger.warning(msg)
            return False, msg

        try:
            raw_bytes = image_path.read_bytes()
        except Exception as e:
            msg = f"读取图片文件失败: {e}"
            logger.error(msg)
            return False, msg

        ext = image_path.suffix.lower()
        mime_type = _MIME_MAP.get(ext, "image/png")
        b64_data = base64.b64encode(raw_bytes).decode("ascii")
        base64_data = f"data:{mime_type};base64,{b64_data}"

        # ── 4. 使用专用提示词模板调用 VLM ──
        prompt_text = await self._build_prompt(focus_area)

        raw_result = await self._call_vlm(base64_data, prompt_text)
        if not raw_result:
            msg = "VLM 识别失败，返回空描述"
            logger.warning(msg)
            return False, msg

        # ── 5. 解析 VLM 返回的结构化 JSON ──
        parsed = self._parse_vlm_json(raw_result)
        result_focus_area = parsed.get("focus_area", focus_area)
        result_description = parsed.get("description", raw_result)
        result_objects = parsed.get("objects", [])
        result_confidence = parsed.get("confidence")

        # ── 6. 获取模型名称 ──
        model_name = self._get_vlm_model_name()

        logger.info(
            f"二次识别完成: image_id={image_id}, focus_area={result_focus_area}, "
            f"model={model_name}, confidence={result_confidence}, "
            f"desc={result_description[:60]}..."
        )
        return True, {
            "image_hash": image_id,
            "focus_area": result_focus_area,
            "description": result_description,
            "objects": result_objects,
            "confidence": result_confidence,
            "model": model_name,
        }

    # ──────────────────────────────────────────
    # VLM 调用方法
    # ──────────────────────────────────────────

    async def _call_vlm(self, base64_data: str, prompt: str) -> str | None:
        """直接调用 VLM 进行识别，使用自定义提示词。

        Args:
            base64_data: base64 编码的图片数据（含 data: 前缀）
            prompt: 提示词文本

        Returns:
            VLM 返回的文本，失败返回 None
        """
        try:
            vlm_model_set = get_model_set_by_task("vlm")
            if not vlm_model_set:
                logger.debug("VLM 模型不可用")
                return None

            context_manager = LLMContextManager()
            request = create_llm_request(
                vlm_model_set,
                "secondary_recognition",
                context_manager=context_manager,
            )

            request.add_payload(LLMPayload(ROLE.USER, [Text(prompt), Image(base64_data)]))
            response = await request.send(stream=False)
            await response

            description = response.message.strip() if response.message else ""
            return description if description else None

        except Exception as e:
            logger.error(f"VLM 识别失败: {e}", exc_info=True)
            return None

    @staticmethod
    def _get_vlm_model_name() -> str:
        """获取当前 VLM 模型名称。"""
        try:
            vlm_model_set = get_model_set_by_task("vlm")
            if vlm_model_set:
                model_entry = vlm_model_set[0]
                return str(model_entry.get("model_identifier", "unknown"))
        except Exception:
            pass
        return "unknown"

    # ──────────────────────────────────────────
    # 混合定位方法
    # ──────────────────────────────────────────

    async def _locate_image(
        self,
        image_hash: str | None = None,
        image_index: int | None = None,
        image_description: str | None = None,
        scope: str = "current_stream",
    ) -> str | None:
        """混合定位图片，按优先级：image_hash > image_index > image_description。

        Args:
            image_hash: 图片哈希值（精确匹配）
            image_index: 图片索引（从1开始，按时间倒序）
            image_description: 描述关键词（模糊匹配）
            scope: 查询范围（current_stream / all_streams）

        Returns:
            image_id 或 None
        """
        # 优先级1：精确哈希匹配
        if image_hash:
            found = await self._locate_image_by_hash(image_hash)
            if found:
                return found

        # 优先级2：索引定位
        if image_index is not None:
            found = await self._locate_image_by_index(image_index, scope=scope)
            if found:
                return found

        # 优先级3：关键词匹配
        if image_description:
            found = await self._locate_image_by_description(image_description)
            if found:
                return found

        return None

    async def _locate_image_by_hash(self, image_hash: str) -> str | None:
        """通过哈希值精确匹配图片。

        Args:
            image_hash: 图片哈希值

        Returns:
            image_id 或 None
        """
        result = await (
            QueryBuilder(Images)
            .filter(image_id=image_hash)
            .order_by("-timestamp")
            .first()
        )
        return result.image_id if result else None

    async def _locate_image_by_index(self, image_index: int, scope: str = "current_stream") -> str | None:
        """通过索引定位图片（从1开始，按时间倒序，1=最新）。

        从 Messages 表中查询 message_type="image" 的记录，
        解析 content 中的 media[0].image_id 获取图片哈希值。

        Args:
            image_index: 图片索引（从1开始，1=最新）
            scope: 查询范围（current_stream 仅查询当前流 / all_streams 全局）

        Returns:
            image_id 或 None
        """
        if image_index < 1:
            return None

        stream_id = self.get_current_stream_id() or "default"

        query = QueryBuilder(Messages).filter(message_type="image")

        if scope == "current_stream":
            query = query.filter(stream_id=stream_id)

        result = await (
            query
            .order_by("-time")
            .offset(image_index - 1)
            .limit(1)
            .first()
        )
        if result is None:
            return None

        # 解析 content 字段提取 image_id
        # content 格式: str({"media": [{"type": "image", "image_id": "hash", "data": "..."}]})
        try:
            content_data = ast.literal_eval(result.content)
            media_list = content_data.get("media", [])
            if media_list:
                return media_list[0].get("image_id")
        except (ValueError, SyntaxError, KeyError, IndexError, AttributeError, TypeError):
            content_preview = getattr(result, 'content', '<no content>')
            if isinstance(content_preview, str):
                content_preview = content_preview[:100]
            logger.warning(f"解析 Messages.content 失败: image_index={image_index}, content={content_preview}...")

        return None

    async def _locate_image_by_description(self, image_description: str) -> str | None:
        """通过关键词模糊匹配定位图片。

        从 Images 表中模糊匹配 description 字段。

        Args:
            image_description: 描述关键词

        Returns:
            image_id 或 None
        """
        if not image_description:
            return None

        result = await (
            QueryBuilder(Images)
            .filter(description__like=f"%{image_description}%")
            .order_by("-timestamp")
            .first()
        )
        return result.image_id if result else None

    # ──────────────────────────────────────────
    # 提示词构建方法
    # ──────────────────────────────────────────

    async def _build_prompt(self, focus_area: str) -> str:
        """使用注册的提示词模板构建 VLM 提示词。

        Args:
            focus_area: 关注区域描述

        Returns:
            渲染后的提示词字符串
        """
        manager = get_prompt_manager()
        tmpl = manager.get_template("image_reidentifier.secondary_recognition")

        if tmpl is not None:
            prompt = await tmpl.set("focus_area", focus_area).build()
            return prompt

        # 降级：模板未注册时使用内联提示词
        logger.warning("提示词模板未注册，使用内联提示词")
        return (
            f"请对这张图片进行详细的二次识别分析。\n\n"
            f"关注区域：{focus_area}\n\n"
            f"请返回以下JSON格式的结果：\n"
            f'{{"focus_area": "你实际分析的区域描述", '
            f'"description": "对该区域的详细描述", '
            f'"objects": ["识别到的对象1", "识别到的对象2"], '
            f'"confidence": 0.95}}\n\n'
            f"请只返回JSON，不要包含其他文字说明。"
        )

    # ──────────────────────────────────────────
    # JSON 解析方法
    # ──────────────────────────────────────────

    @staticmethod
    def _parse_vlm_json(raw: str) -> dict:
        """解析 VLM 返回的 JSON 结果。

        VLM 可能返回包含 markdown 代码块的 JSON，需要提取。

        Args:
            raw: VLM 原始返回文本

        Returns:
            解析后的字典，解析失败时返回空字典
        """
        if not raw:
            return {}

        # 尝试提取 markdown 代码块中的 JSON
        text = raw.strip()
        if text.startswith("```"):
            # 去除 ```json 和 ``` 包裹
            lines = text.split("\n")
            # 去除首行（```json 或 ```）
            lines = lines[1:] if lines else []
            # 去除末行（```）
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # 尝试从文本中提取第一个 JSON 对象
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(text[start : end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        logger.warning(f"无法解析VLM返回的JSON: {raw[:100]}...")
        return {}
