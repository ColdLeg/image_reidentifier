"""image_reidentifier 工具包。

导出插件提供的所有 Tool 组件，供插件系统自动发现和注册。
"""

from .secondary_recognition import SecondaryRecognitionTool

__all__ = [
    "SecondaryRecognitionTool",
]