import torch
import numpy as np
from PIL import Image
import io
import time
import random
from google import genai
from google.genai import types
import comfy.model_management
import re

def get_retry_wait_info(error_str, retry_count, default_wait=15):
    """Calculates backoff wait time and message based on error type and retry count."""
    if not isinstance(error_str, str):
        error_str = str(error_str)
        
    err_str_lower = error_str.lower()
    
    if "503" in err_str_lower or "overloaded" in err_str_lower:
        wait_time = min(60.0, (2 ** (retry_count - 1)) + random.uniform(0, 1))
        msg = f"⏳ 503 Overload | Exponential backoff: Waiting {wait_time:.2f}s..."
        return wait_time, msg
        
    if "429" in err_str_lower or "quota exhausted" in err_str_lower or "rate limit" in err_str_lower:
        wait_time = 60.0
        msg = f"⏳ Rate Limit Cooldown | Waiting 60s for per-minute quota refresh..."
        return wait_time, msg
        
    if "500" in err_str_lower or "internal error" in err_str_lower:
        wait_time = random.uniform(5, 10)
        msg = f"⏳ 500 Internal Error | Waiting {wait_time:.2f}s before retry..."
        return wait_time, msg

    match = re.search(r"Please retry in (\d+\.?\d*)s", error_str)
    if match:
        try:
            wait_time = float(match.group(1))
            msg = f"⏳ Dynamic Cooldown | Waiting {wait_time}s (Google request) before retry..."
            return wait_time, msg
        except:
            pass

    return float(default_wait), f"⏳ Buffer Wait | Waiting {default_wait}s to clear API congestion..."

def clean_error_msg(error_str):
    """Makes API errors more readable for the console."""
    err_str_lower = str(error_str).lower()
    if "503" in err_str_lower or "overloaded" in err_str_lower:
        return "Model Overloaded (503) - Server capacity reached. Retrying with backoff..."
    if "429" in err_str_lower or "quota exhausted" in err_str_lower or "rate limit" in err_str_lower:
        return "Rate Limit/Quota Exhausted (429). Waiting for rolling 60s refresh..."
    if "404" in err_str_lower or "not found" in err_str_lower:
        return f"Not Found/Model Unavailable (404). Check model name or region. Detail: {str(error_str)[:100]}"
    if "403" in err_str_lower or "permission denied" in err_str_lower:
        return f"Permission Denied (403). Check API key and billing. Detail: {str(error_str)[:100]}"
    if "400" in err_str_lower or "invalid argument" in err_str_lower:
         return f"Invalid Request (400). Check arguments (e.g., inlineData). Detail: {str(error_str)[:100]}"
    if "500" in err_str_lower or "internal error" in err_str_lower:
         return f"Internal Server Error (500). Google side issue. Retrying shortly..."

    clean = re.sub(r"\{'error': \{.*?\}\}", "", str(error_str)).strip()
    return clean if clean else str(error_str)

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
    - Enhanced English logging with Batch counter support.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system_instruction": ("STRING", {"multiline": True, "default": "You are a professional image generator. Maintain high cinematic quality.", "placeholder": "System instructions..."}),
                "prompt": ("STRING", {"multiline": True, "default": "Make this image cyberpunk style", "placeholder": "Enter your prompt here..."}),
                "model": (["gemini-3.1-pro-preview", "gemini-3-pro-image-preview", "gemini-2.0-flash", "gemini-2.0-pro-exp-02-05"], {"default": "gemini-3-pro-image-preview"}),
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

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "text_response")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "process_batch"
    CATEGORY = "Gemini AI"

    def process_batch(self, system_instruction, prompt, model, seed, aspect_ratio, resolution, response_modalities, api_key_1, api_key_2, api_key_3, api_max_retries, images=None):
        # Determine total items to process based on inputs
        batch_size = max(len(prompt), len(images) if images is not None else 0)
        
        results_images = []
        results_texts = []
        
        # Build available keys list for load balancing
        available_keys = [k.strip() for k in [api_key_1[0] if api_key_1 else "", 
                                              api_key_2[0] if api_key_2 else "", 
                                              api_key_3[0] if api_key_3 else ""] if k and k.strip()]

        for i in range(batch_size):
            # Intra-batch pacing delay to avoid instant 429 on large batches 
            if i > 0:
                print(f"   ⏳ Throttling: Waiting 2.5s")
                time.sleep(2.5)

            # Resolve parameters for this specific item in the batch
            # We use modulo to cycle through parameters if they are shorter than the batch size (standard ComfyUI behavior)
            curr_sys = system_instruction[i % len(system_instruction)]
            curr_prompt = prompt[i % len(prompt)]
            curr_model = model[i % len(model)]
            
            # Smart seed logic: if we have one seed input but multiple prompts, auto-increment
            if len(seed) == 1:
                curr_seed = (seed[0] + i) % 0xffffffffffffffff
            else:
                curr_seed = seed[i % len(seed)]
                
            curr_aspect = aspect_ratio[i % len(aspect_ratio)]
            curr_res = resolution[i % len(resolution)]
            curr_modal = response_modalities[i % len(response_modalities)]
            
            # API Keys and Global Settings
            # Instead of passing individual keys, we will pass them all but tell the exec function
            # which one to try FIRST based on round-robin logic.
            k1 = api_key_1[0] if api_key_1 else ""
            k2 = api_key_2[0] if api_key_2 else ""
            k3 = api_key_3[0] if api_key_3 else ""
            max_r = api_max_retries[0] if api_max_retries else 10
            
            # Key rotation starting index for load-balancing
            round_robin_idx = (i % len(available_keys)) if available_keys else 0
            
            curr_img_input = images[i % len(images)] if images is not None else None

            # Execute single generation
            out_img, out_txt = self._exec_single(
                curr_sys, curr_prompt, curr_model, curr_seed, curr_aspect, curr_res, curr_modal,
                k1, k2, k3, max_r, 
                curr_img_input, 
                curr_batch=i+1, 
                total_batch=batch_size,
                start_key_idx=round_robin_idx
            )
            
            results_images.append(out_img)
            results_texts.append(out_txt)

        return (results_images, results_texts)

    def _exec_single(self, system_instruction, prompt, model, seed, aspect_ratio, resolution, response_modalities, api_key_1, api_key_2, api_key_3, api_max_retries, images=None, curr_batch=1, total_batch=1, start_key_idx=0):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required!")

        all_keys = [k.strip() for k in [api_key_1, api_key_2, api_key_3] if k and k.strip()]
        if not all_keys:
            raise ValueError("No API Key provided!")
            
        # Re-order keys to start with the load-balanced index first, followed by the others naturally
        keys = all_keys[start_key_idx:] + all_keys[:start_key_idx]

        image_parts = []
        if images is not None:
            # images can be [H,W,C] or [B,H,W,C]
            img_to_proc = images if len(images.shape) == 4 else images.unsqueeze(0)
            for j in range(img_to_proc.shape[0]):
                img_tensor = img_to_proc[j]
                np_img = 255. * img_tensor.cpu().numpy()
                pil_img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                image_parts.append(types.Part.from_bytes(data=buffered.getvalue(), mime_type="image/png"))

        def call_api(current_key):
            client = genai.Client(
                api_key=current_key,
                http_options={
                    "timeout": 300000,
                    "client_args": {"http2": False}
                }
            )

            actual_model = model
            effective_system = GEMINI_IMAGE_SYS_PROMPT
            if system_instruction and system_instruction.strip():
                effective_system = system_instruction.strip() + "\n\n" + GEMINI_IMAGE_SYS_PROMPT

            modalities_list = ["IMAGE"] if response_modalities == "IMAGE" else ["IMAGE", "TEXT"]
            no_aspect_ratio_models = {"gemini-2.0-flash", "gemini-2.0-pro-exp-02-05", "gemini-3.1-pro-preview"}
            
            safety_settings = [
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
            ]

            config_kwargs = {
                "system_instruction": effective_system,
                "response_modalities": modalities_list,
                "image_config": {"image_size": resolution},
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
                "safety_settings": safety_settings,
            }
            if aspect_ratio != "auto" and actual_model not in no_aspect_ratio_models:
                config_kwargs["image_config"]["aspect_ratio"] = aspect_ratio

            TEXT_ONLY_RETRIES = 3
            for text_retry in range(TEXT_ONLY_RETRIES):
                # Informative logging with Batch counter
                batch_info = f" | Batch {curr_batch}/{total_batch}" if total_batch > 1 else ""
                print(f"\n🎨 Gemini Image{batch_info} | Generating image...")
                print(f"   Model: {actual_model} | Resolution: {resolution} | Ratio: {aspect_ratio} | Seed: {seed}")
                print(f"   Key:   Using Key #{text_retry + 1} (****{current_key[-4:]})")
                if text_retry > 0:
                    print(f"   ↻ Safety Retry {text_retry}/{TEXT_ONLY_RETRIES - 1} (Retrying same key...)")

                content_parts = [types.Part.from_text(text=prompt)] + image_parts
                response_stream = client.models.generate_content_stream(
                    model=actual_model,
                    contents=[types.Content(role="user", parts=content_parts)],
                    config=types.GenerateContentConfig(**config_kwargs),
                )

                out_tensor = None
                full_text = ""
                image_found = False
                chunk_count = 0
                last_finish_reason = None
                for chunk in response_stream:
                    chunk_count += 1
                    if comfy.model_management.processing_interrupted():
                        raise Exception("Interrupted by user")
                    if not chunk.candidates or not chunk.candidates[0].content:
                        if chunk.candidates and chunk.candidates[0].finish_reason:
                             last_finish_reason = chunk.candidates[0].finish_reason
                        continue
                        
                    last_finish_reason = chunk.candidates[0].finish_reason
                    if not chunk.candidates[0].content.parts:
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

                if last_finish_reason == types.FinishReason.IMAGE_OTHER:
                     raise ValueError(f"IMAGE_OTHER: Image generation blocked by Google's policy. Try a more generic prompt.")
                elif last_finish_reason not in [types.FinishReason.STOP, types.FinishReason.MAX_TOKENS]:
                     raise ValueError(f"Stream interrupted. Reason: {last_finish_reason}. Chunks: {chunk_count}")

                if image_found:
                    print(f"   ✅ Success! Image received ({chunk_count} chunks handled).")
                    return out_tensor, full_text

                print(f"   ⚠️ Warning: Model returned text description instead of image.")
                if text_retry < TEXT_ONLY_RETRIES - 1:
                    for _ in range(5):
                        if comfy.model_management.processing_interrupted():
                            raise Exception("Interrupted by user")
                        time.sleep(1)

            raise ValueError(f"Model returned text only after {TEXT_ONLY_RETRIES} attempts.")

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
                    print(f"\n❌ Key Failure | Key #{index + 1} (****{key[-4:]}) failed.")
                    print(f"   Reason: {clean_error_msg(error_str)}")
                    
                    if "disconnected" in error_str.lower() or "remote protocol" in error_str.lower() or "10054" in error_str:
                        print(f"   ℹ️ Notice: Transient network hiccup. Automatic retry in progress...")

                    if "400 INVALID_ARGUMENT" in error_str:
                        raise ValueError(f"Fatal error (not retryable): {error_str}")
                    if index < len(keys) - 1:
                        print(f"🔄 Rotating API keys to find an available worker...")
            
            retry_count += 1
            if retry_count < api_max_retries:
                current_wait_sec, wait_msg = get_retry_wait_info(error_str if 'error_str' in locals() else "", retry_count, default_wait=wait_time)
                
                print(f"⚠️ Round {retry_count}/{api_max_retries} Exhausted | All keys temporarily unavailable.")
                print(wait_msg)
                
                for _ in range(int(current_wait_sec) + 1):
                    if comfy.model_management.processing_interrupted():
                        raise Exception("Interrupted by user")
                    time.sleep(1)
        
        raise ValueError(f"Failed after {api_max_retries} rounds.")

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
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "model": (["gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.0-pro-exp-02-05", "gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-pro", "gemini-1.5-flash"], {"default": "gemini-3.1-pro-preview"}),
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
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_text"
    CATEGORY = "Gemini API"

    def generate_text(self, system_instruction, user_prompt, model, seed, max_output_tokens, api_key_1, api_key_2, api_key_3, api_max_retries, batch_count, images=None):
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

        results = []
        
        # Build available keys list for load balancing
        available_keys = [k.strip() for k in [api_key_1, api_key_2, api_key_3] if k and k.strip()]

        for b in range(batch_count):
            if b > 0:
                print(f"   ⏳ Throttling: Waiting 2.5s")
                time.sleep(2.5)

            current_seed = (seed + b) % 2147483647
            # Load balancing offset for API keys
            start_key_idx = b % len(available_keys)
            keys = available_keys[start_key_idx:] + available_keys[:start_key_idx]
            
            def call_api(current_key):
                nonlocal current_seed
                client = genai.Client(
                    api_key=current_key,
                    http_options={
                        "timeout": 300000,
                        "client_args": {"http2": False}
                    }
                )
                config_kwargs = {
                    "system_instruction": system_instruction,
                    "seed": current_seed,
                    "response_modalities": ["TEXT"],
                    "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
                    "safety_settings": [
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                    ],
                }
                if max_output_tokens > 0:
                    config_kwargs["max_output_tokens"] = max_output_tokens
                
                batch_str = f" | Batch {b+1}/{batch_count}" if batch_count > 1 else ""
                print(f"\n📝 Gemini Prompt{batch_info} | Processing..." if 'batch_info' in locals() else f"\n📝 Gemini Prompt{batch_str} | Processing...")
                print(f"   Model: {model} | Seed: {current_seed}")
                
                response_stream = client.models.generate_content_stream(
                    model=model,
                    contents=[types.Content(role="user", parts=input_parts)],
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                
                full_text = ""
                chunk_count = 0
                last_finish_reason = None
                for chunk in response_stream:
                    if comfy.model_management.processing_interrupted():
                        raise Exception("Interrupted by user")
                    if not chunk.candidates:
                        continue
                    
                    last_finish_reason = chunk.candidates[0].finish_reason
                    if not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                        continue
                        
                    for part in chunk.candidates[0].content.parts:
                        if part.text:
                            full_text += part.text
                            chunk_count += 1
                
                if last_finish_reason not in [types.FinishReason.STOP, types.FinishReason.MAX_TOKENS]:
                     raise ValueError(f"Stream interrupted. Reason: {last_finish_reason}")
                
                if not full_text:
                    raise ValueError("Model returned empty text.")
                
                print(f"   ✅ Complete | Received {len(full_text)} characters in {chunk_count} chunks.")
                return full_text

            # Strategy: execute this batch item with full key rotation/retry
            retry_count = 0
            wait_time = 5
            success = False
            item_text = ""
            
            while retry_count < api_max_retries:
                for index, key in enumerate(keys):
                    if comfy.model_management.processing_interrupted():
                        raise Exception("Interrupted by user")
                    try:
                        item_text = call_api(key)
                        success = True
                        break
                    except Exception as e:
                        error_str = str(e)
                        print(f"\n❌ Key Failure | Key #{index + 1} (****{key[-4:]}) failed.")
                        print(f"   Reason: {clean_error_msg(error_str)}")
                        if "400 INVALID_ARGUMENT" in error_str:
                            raise ValueError(f"Fatal: {error_str}")
                
                if success:
                    break
                
                retry_count += 1
                if retry_count < api_max_retries:
                    current_wait_sec, wait_msg = get_retry_wait_info(error_str if 'error_str' in locals() else "", retry_count, default_wait=wait_time)
                    print(f"⚠️ Round {retry_count}/{api_max_retries} Exhausted | {wait_msg}")
                    for _ in range(int(current_wait_sec) + 1):
                        if comfy.model_management.processing_interrupted():
                            raise Exception("Interrupted by user")
                        time.sleep(1)
            
            if success:
                results.append(item_text)
            else:
                results.append(f"ERROR: Batch {b+1} failed after all retries.")

        return (results,)
