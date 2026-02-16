# 🚀 ComfyUI-Gemini3-API-Fallback

Expertly designed custom nodes for **Google Gemini** integration in ComfyUI, featuring a robust multi-key retry system to eliminate server errors and rate limits.

---

## 🌟 Key Features

### 🛡️ 3-Key Iterative System (API Fallback)
Stop worrying about "Server Error 500" or rate limits. These nodes allow you to input up to **3 different API keys**. If the primary key fails, the system automatically rotates to the next one and retries up to your configured `max_retries`.

### 🎨 Gemini 3 Pro Image (API Fallback)
Generate stunning high-fidelity images directly from ComfyUI using the latest `gemini-3-pro-image-preview` model.
- **Native Resolutions**: 1K, 2K, and 4K support.
- **Aspect Ratio Control**: Auto, 1:1, 16:9, 9:16, 21:9, and more.
- **Visual Context**: Support for image inputs to guide generation (Multimodal).

### 📝 Gemini 3 Pro (API Fallback)
The ultimate tool for prompt engineering, image analysis, and creative writing.
- **Versatile Models**: Includes `gemini-3-pro-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, and specialized previews.
- **Image-to-Text**: Upload images to get detailed descriptions or use them as a reference for new prompts.
- **Clean UI**: Streamlined parameters focused on results.

---

---

## 🔑 Crucial: Setting up your Google API Keys

To use **Gemini-3-Pro-Image** models, you **must** enable billing in your Google Cloud project and verify your quotas. Otherwise, you will encounter a `Limit: 0` error.

### ⚠️ Troubleshooting "Limit: 0" or "429 RESOURCE_EXHAUSTED"
If you get an error saying `limit: 0`, even with billing enabled:
1. **Model Quota**: Image models (`gemini-2.0-flash` or `imagen`) have a separate quota. Go to [Google Cloud Console > Quotas](https://console.cloud.google.com/iam-admin/quotas) and search for `generativelanguage.googleapis.com/generate_requests_per_model_per_day`.
2. **Project Tier**: Return to [Google AI Studio](https://aistudio.google.com/), go to **Settings > Plan**, and confirm your project is on the **"Pay-as-you-go"** plan. If it says "Free", click "Edit" and choose your Billing project.
3. **Region**: Some image models are only available in certain regions. Ensure your project is not restricted.

### 1. Get your API Keys
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google Account.
3. Click on the **"Get API key"** button on the top left sidebar.
4. Click **"Create API key in new project"**.
5. Copy your API Key and repeat to get up to 3 keys if you want full fallback protection.

### 2. Enable Billing (Required for Image Generation)
By enabling billing, you move to the "Pay-as-you-go" tier which unlocks the quota for image models. 
> **Note:** Google often provides $300 in free credits for new accounts, which covers a significant amount of usage.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Select your project from the top dropdown.
3. Navigate to **Billing** in the sidebar menu.
4. Click **"Add billing account"** and enter your payment information.
5. **Link your project**: In the "Account Management" tab of the Billing section, ensure your project is linked to the new billing account.
6. **Set Budget Alerts (Highly Recommended)**: In the "Budgets & Alerts" menu, create a budget (e.g., $1 or $5) to receive email notifications and avoid unexpected charges.

### 3. Verify in AI Studio
Return to [Google AI Studio](https://aistudio.google.com/), go to **Settings/Plan**, and confirm your project is now on the **"Pay-as-you-go"** plan. This will replace the "Limit: 0" with a functional quota.

---

## 🛠️ Installation

1. Navigate to your ComfyUI `custom_nodes` folder.
2. Clone this repository:
   ```bash
   git clone https://github.com/USER/ComfyUI-Gemini3-API-Fallback
   ```
3. Install dependencies:
   ```bash
   python -m pip install google-genai pillow numpy torch
   ```

---

## ⚙️ Configurable Parameters
- `api_key_1, 2, 3`: Your iterative keys.
- `max_retries`: Number of full rotation rounds to attempt before giving up (Default: 10).
- `English Logging`: Real-time console feedback on key rotation and server wait times.

---