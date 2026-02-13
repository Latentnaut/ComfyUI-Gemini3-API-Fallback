from .gemini_node import Gemini3ProImageGenNode, GeminiPromptGenerator

NODE_CLASS_MAPPINGS = {
    "Gemini3ProImageGenNode": Gemini3ProImageGenNode,
    "GeminiPromptGenerator": GeminiPromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini3ProImageGenNode": "🤖 Gemini 3 Pro Image (API Fallback)",
    "GeminiPromptGenerator": "🤖 Gemini 3 Pro (API Fallback)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
