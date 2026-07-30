"""image_reidentifier 插件入口。

插件加载后注册 VLM 二次识别的提示词模板。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.core.components import BasePlugin, register_plugin
from src.core.prompt import get_prompt_manager, trim
from src.kernel.logger import get_logger

from .config import ImageReidentifierConfig
from .tools.secondary_recognition import SecondaryRecognitionTool

logger = get_logger("image_reidentifier")


@register_plugin
class ImageReidentifierPlugin(BasePlugin):
    """image_reidentifier 插件。"""

    plugin_name: str = "image_reidentifier"
    plugin_description: str = "对已识别图片进行二次识别，通过VLM生成更精准的描述"
    plugin_version: str = "1.0.0"

    configs: list[type] = [ImageReidentifierConfig]
    dependent_components: list[str] = []

    def __init__(self, config: ImageReidentifierConfig | None = None) -> None:
        super().__init__(config)

    def get_components(self) -> list[type]:
        """返回本插件提供的组件类。"""
        return [SecondaryRecognitionTool]

    async def on_plugin_loaded(self) -> None:
        """插件加载完成后注册提示词模板。"""
        await self._register_prompt_template()

    async def on_plugin_unloaded(self) -> None:
        """插件卸载前注销提示词模板。"""
        try:
            manager = get_prompt_manager()
            manager.unregister_template("image_reidentifier.secondary_recognition")
        except Exception:
            pass

    async def _register_prompt_template(self) -> None:
        """从 YAML 文件加载提示词模板并注册到 PromptManager。"""
        yaml_path = Path(__file__).parent / "prompts" / "secondary_recognition.yaml"
        if not yaml_path.exists():
            logger.warning(f"提示词模板文件不存在: {yaml_path}")
            return

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            template_str = data.get("template", "")
            if not template_str:
                logger.warning("提示词模板内容为空，跳过注册")
                return

            manager = get_prompt_manager()
            manager.get_or_create(
                name="image_reidentifier.secondary_recognition",
                template=template_str,
                policies={
                    "focus_area": trim(),
                },
            )
            logger.info("提示词模板已注册: image_reidentifier.secondary_recognition")
        except Exception as e:
            logger.warning(f"注册提示词模板失败: {e}")