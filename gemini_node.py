import torch
import numpy as np
from PIL import Image
import io
import time
from google import genai
from google.genai import types
import comfy.model_management

# Official ComfyUI system prompt for Gemini image generation
GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of "
    "format, intent, or abstraction—as literal visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, "
    "you must creatively invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
)

class Gemini3ProImageGenNode:
    """
    ComfyUI Node for Google Gemini 3 Pro (Image Generation).
    Features:
    - Supports gemini-3-pro-image-preview.
    - 3-Key Iterative Retry System with configurable rounds.
    - Enhanced English logging.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system_instruction": ("STRING", {"multiline": True, "default": "You are a professional image generator. Maintain high cinematic quality.", "placeholder": "System instructions..."}),
                "prompt": ("STRING", {"multiline": True, "default": "Make this image cyberpunk style", "placeholder": "Enter your prompt here..."}),
                "model": (["gemini-3-pro-image-preview", "gemini-3-pro-preview", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"], {"default": "gemini-3-pro-image-preview"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                
                "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "auto"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "response_modalities": (["IMAGE+TEXT", "IMAGE"], {"default": "IMAGE+TEXT"}),

                "api_key_1": ("STRING", {"multiline": False, "default": "", "placeholder": "Primary API Key (Required)"}),
                "api_key_2": ("STRING", {"multiline": False, "default": "", "placeholder": "Backup API Key 1 (Optional)"}),
                "api_key_3": ("STRING", {"multiline": False, "default": "", "placeholder": "Backup API Key 2 (Optional)"}),
                "api_max_retries": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "text_response")
    FUNCTION = "process_image"
    CATEGORY = "Gemini AI"

    def process_image(self, system_instruction, prompt, model, seed, aspect_ratio, resolution, response_modalities, api_key_1, api_key_2, api_key_3, api_max_retries, images=None):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required!")

        keys = [k.strip() for k in [api_key_1, api_key_2, api_key_3] if k and k.strip()]
        if not keys:
            raise ValueError("No API Key provided!")

        image_parts = []
        if images is not None:
            for i in range(images.shape[0]):
                img_tensor = images[i]
                np_img = 255. * img_tensor.cpu().numpy()
                pil_img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                image_parts.append(types.Part.from_bytes(data=buffered.getvalue(), mime_type="image/png"))

        def call_api(current_key):
            # Force REST by disabling HTTP/2 in httpx and setting a long timeout
            client = genai.Client(
                api_key=current_key,
                http_options={
                    "timeout": 300000,  # 300 seconds (5 minutes)
                    "client_args": {"http2": False}
                }
            )

            # Model name mapping for compatibility
            model_map = {
                "gemini-3-pro-image-preview": "gemini-2.0-flash",
                "gemini-2.5-pro": "gemini-2.0-pro-exp-02-05",
                "gemini-2.5-flash": "gemini-2.0-flash",
            }
            actual_model = model_map.get(model, model)

            # Use the official image generation system prompt, appended to user's system instruction
            effective_system = GEMINI_IMAGE_SYS_PROMPT
            if system_instruction and system_instruction.strip():
                effective_system = system_instruction.strip() + "\n\n" + GEMINI_IMAGE_SYS_PROMPT

            # Match official node: IMAGE -> ["IMAGE"], IMAGE+TEXT -> ["IMAGE", "TEXT"]
            if response_modalities == "IMAGE":
                modalities_list = ["IMAGE"]
            else:
                modalities_list = ["IMAGE", "TEXT"]

            # Models that do NOT support aspect_ratio
            no_aspect_ratio_models = {"gemini-3-pro-preview", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"}

            # Build config (matching working backup: dict-based image_config, no seed)
            config_kwargs = {
                "system_instruction": effective_system,
                "response_modalities": modalities_list,
                # Disable AFC (Automatic Function Calling) to reduce overhead
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
                "image_config": {"image_size": resolution}
            }
            if aspect_ratio != "auto" and actual_model not in no_aspect_ratio_models:
                config_kwargs["image_config"]["aspect_ratio"] = aspect_ratio

            # Retry text-only responses per key (model behavior, not key issue)
            TEXT_ONLY_RETRIES = 3
            for text_retry in range(TEXT_ONLY_RETRIES):
                print(f"\n🔵 [Gemini Image Gen] Sending request...")
                print(f"   Model: {actual_model} (mapped from {model})")
                print(f"   Key:   ****{current_key[-4:]}")
                print(f"   Config: modalities={modalities_list}, image_config={config_kwargs.get('image_config', 'none')}")
                if text_retry > 0:
                    print(f"   ↻ Text-only retry {text_retry}/{TEXT_ONLY_RETRIES - 1} (same key)")

                content_parts = [types.Part.from_text(text=prompt)] + image_parts
                response_stream = client.models.generate_content_stream(
                    model=actual_model,
                    contents=[types.Content(role="user", parts=content_parts)],
                    config=types.GenerateContentConfig(**config_kwargs),
                )

                out_tensor = None
                full_text = ""
                image_found = False
                for chunk in response_stream:
                    if comfy.model_management.processing_interrupted():
                        raise Exception("Interrupted by user")
                    if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                        continue
                    for part in chunk.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.data:
                            image_data = part.inline_data.data
                            out_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
                            out_np = np.array(out_pil).astype(np.float32) / 255.0
                            out_tensor = torch.from_numpy(out_np)[None,]
                            image_found = True
                        if part.text:
                            full_text += part.text

                if image_found:
                    return out_tensor, full_text

                # Text-only response — retry with same key after short delay
                print(f"   ⚠️ Model returned text only (attempt {text_retry + 1}/{TEXT_ONLY_RETRIES})")
                if text_retry < TEXT_ONLY_RETRIES - 1:
                    # Short sleep before retrying (model non-determinism, not rate limit)
                    for _ in range(5):
                        if comfy.model_management.processing_interrupted():
                            raise Exception("Interrupted by user")
                        time.sleep(1)

            raise ValueError(f"Model returned text only after {TEXT_ONLY_RETRIES} attempts: {full_text[:200]}...")

        retry_count = 0
        wait_time = 15
        while retry_count < api_max_retries:
            for index, key in enumerate(keys):
                if comfy.model_management.processing_interrupted():
                    raise Exception("Interrupted by user")
                try:
                    return call_api(key)
                except Exception as e:
                    if comfy.model_management.processing_interrupted():
                        raise Exception("Interrupted by user")
                    error_str = str(e)
                    print(f"\n❌ [Gemini Image Gen] Key #{index + 1} (****{key[-4:]}) failed.")
                    print(f"   Error: {error_str}")
                    # Bail immediately on 400 errors (deterministic, will never succeed on retry)
                    if "400 INVALID_ARGUMENT" in error_str:
                        raise ValueError(f"Gemini Image Gen fatal error (not retryable): {error_str}")
                    if index < len(keys) - 1:
                        print(f"🔄 [Gemini Image Gen] Rotating to next backup key...")
            
            retry_count += 1
            if retry_count < api_max_retries:
                print(f"\n⚠️ [Gemini Image Gen] All keys failed in Round {retry_count}/{api_max_retries}.")
                print(f"⏳ [Gemini Image Gen] Waiting {wait_time}s to avoid rate limits...\n")
                
                # Sleep in small increments to allow interrupt
                for _ in range(wait_time):
                    if comfy.model_management.processing_interrupted():
                        raise Exception("Interrupted by user")
                    time.sleep(1)
        
        raise ValueError(f"Gemini Image Gen failed after {api_max_retries} rounds.")

class GeminiPromptGenerator:
    """
    ComfyUI Node for Google Gemini (Text/Prompt Generation).
    Features:
    - Supports multiple text models.
    - 3-Key Iterative Retry System with configurable rounds.
    - Enhanced English logging.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system_instruction": ("STRING", {"multiline": True, "default": "You are a creative writer. Describe the image in detail.", "placeholder": "System instructions..."}),
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe this image", "placeholder": "User prompt..."}),
                "model": (["gemini-3-pro-preview", "gemini-2.0-pro-exp-02-05", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-lite-preview-02-05"], {"default": "gemini-3-pro-preview"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                
                "max_output_tokens": ("INT", {"default": 0, "min": 0, "max": 128000, "step": 64}),

                "api_key_1": ("STRING", {"multiline": False, "default": "", "placeholder": "Primary API Key (Required)"}),
                "api_key_2": ("STRING", {"multiline": False, "default": "", "placeholder": "Backup API Key 1 (Optional)"}),
                "api_key_3": ("STRING", {"multiline": False, "default": "", "placeholder": "Backup API Key 2 (Optional)"}),
                "api_max_retries": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "generate_text"
    CATEGORY = "Gemini AI"

    def generate_text(self, system_instruction, user_prompt, model, seed, max_output_tokens, api_key_1, api_key_2, api_key_3, api_max_retries, images=None):
        if not user_prompt and images is None:
             raise ValueError("At least a prompt or an image is required.")

        keys = [k.strip() for k in [api_key_1, api_key_2, api_key_3] if k and k.strip()]
        if not keys:
            raise ValueError("No API Key provided!")

        input_parts = []
        if images is not None:
             for i in range(images.shape[0]):
                img_tensor = images[i]
                np_img = 255. * img_tensor.cpu().numpy()
                pil_img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                input_parts.append(types.Part.from_bytes(data=buffered.getvalue(), mime_type="image/png"))
        
        if user_prompt:
            input_parts.append(types.Part.from_text(text=user_prompt))

        def call_api(current_key):
            # Force REST by disabling HTTP/2 in httpx and setting a long timeout
            # Complex prompts (e.g., multi-angle grids) need much more processing time
            client = genai.Client(
                api_key=current_key,
                http_options={
                    "timeout": 300000,  # 300 seconds (5 minutes) for complex prompts
                    "client_args": {"http2": False}
                }
            )
            # Clamp seed to INT32 range (Gemini API requirement)
            safe_seed = seed % 2147483647
            config_kwargs = {
                "system_instruction": system_instruction,
                "seed": safe_seed,
                "response_modalities": ["TEXT"],
                # Disable AFC (Automatic Function Calling) to reduce overhead
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            }
            if max_output_tokens > 0:
                config_kwargs["max_output_tokens"] = max_output_tokens
                
            # Model name mapping for compatibility/hallucinated names
            model_map = {
                "gemini-3-pro-image-preview": "gemini-2.0-flash",
                "gemini-2.5-pro": "gemini-2.0-pro-exp-02-05",
                "gemini-2.5-flash": "gemini-2.0-flash",
            }
            actual_model = model_map.get(model, model)
                
            print(f"\n🔵 [Gemini Prompt Gen] Sending request (streaming)...")
            print(f"   Model: {actual_model} (mapped from {model})")
            print(f"   Key:   ****{current_key[-4:]}")
            
            # Use streaming to avoid server-side 504 DEADLINE_EXCEEDED
            # Streaming keeps the connection alive as chunks arrive,
            # preventing timeout on long/complex responses
            response_stream = client.models.generate_content_stream(
                model=actual_model,
                contents=[types.Content(role="user", parts=input_parts)],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            
            full_text = ""
            chunk_count = 0
            for chunk in response_stream:
                if comfy.model_management.processing_interrupted():
                    raise Exception("Interrupted by user")
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if part.text:
                            full_text += part.text
                            chunk_count += 1
            
            if not full_text:
                raise ValueError("Model returned empty text.")
            
            print(f"   ✅ Received {chunk_count} chunks, {len(full_text)} characters total.")
            return (full_text,)

        retry_count = 0
        wait_time = 5
        while retry_count < api_max_retries:
            for index, key in enumerate(keys):
                if comfy.model_management.processing_interrupted():
                    raise Exception("Interrupted by user")
                try:
                    return call_api(key)
                except Exception as e:
                    if comfy.model_management.processing_interrupted():
                        raise Exception("Interrupted by user")
                    error_str = str(e)
                    print(f"\n❌ [Gemini Prompt Gen] Key #{index + 1} (****{key[-4:]}) failed.")
                    print(f"   Error: {error_str}")
                    # Bail immediately on 400 errors (deterministic, will never succeed on retry)
                    if "400 INVALID_ARGUMENT" in error_str:
                        raise ValueError(f"Gemini Prompt Gen fatal error (not retryable): {error_str}")
                    if index < len(keys) - 1:
                        print(f"🔄 [Gemini Prompt Gen] Rotating to next backup key...")
            
            retry_count += 1
            if retry_count < api_max_retries:
                print(f"\n⚠️ [Gemini Prompt Gen] All keys failed in Round {retry_count}/{api_max_retries}.")
                print(f"⏳ [Gemini Prompt Gen] Waiting {wait_time}s to avoid rate limits...\n")
                
                # Sleep in small increments to allow interrupt
                for _ in range(wait_time):
                    if comfy.model_management.processing_interrupted():
                        raise Exception("Interrupted by user")
                    time.sleep(1)

        raise ValueError(f"Gemini Prompt Gen failed after {api_max_retries} rounds.")
