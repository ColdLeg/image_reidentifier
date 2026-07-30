"""image_reidentifier 插件配置。

配置文件默认路径：config/plugins/image_reidentifier/config.toml

本插件为纯无状态工具，VLM 模型通过 model.toml 的 vlm 任务配置，
提示词通过 prompts/secondary_recognition.yaml 配置，无需额外插件配置项。
"""

from __future__ import annotations

from typing import ClassVar

from src.core.components.base.config import BaseConfig


class ImageReidentifierConfig(BaseConfig):
    """image_reidentifier 插件配置。"""

    name: ClassVar[str] = "config"
    description: ClassVar[str] = "图片二次识别插件配置"