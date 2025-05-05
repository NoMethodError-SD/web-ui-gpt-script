# GPT Prompts Extension for Stable Diffusion WebUI

This extension integrates OpenAI's GPT models with Stable Diffusion WebUI to generate structured, high-quality prompts for image generation.

## Features

- 🤖 Uses GPT models to generate structured prompts
- 🔄 Batch generation support
- ⚙️ Customizable system prompt and user prompt presets
- 💾 Persistent configuration and prompt presets (JSON files)
- 🎨 Organized prompt structure (freeform description, subject, environment, quality tags)
- ⚡ Async processing for better performance
- 🔑 API key management through settings
- 🎭 Multi-model support with flexible selection modes

## Installation

1. Open the "Extensions" tab in Stable Diffusion WebUI
2. Go to "Install from URL"
3. Paste this repository URL
4. Click "Install"
5. Restart Stable Diffusion WebUI

## Configuration

1. Go to Settings > OpenAI Prompt Generator
2. Enter your OpenAI API key
3. (Optional) Configure:
   - API base URL (for alternative endpoints)
   - LLM Model selection
   - Rate limits
   - Default system prompt
   - Default model
4. All configuration is saved in `scripts/openai_batcher_config.json` for persistence.

## Prompt Presets

The extension supports two types of prompt presets:
- **System Prompt Presets**: Templates for the setup/system prompt (stored in `scripts/system_prompts.json`).
- **User Prompt Presets**: Templates for your main/user prompt (stored in `scripts/user_prompts.json`).

You can load, save, and manage these presets directly from the UI. System presets include a "Default" option that cannot be overwritten. User presets are fully customizable.

## Usage

1. Open the "GPT Prompts" script in txt2img or img2img
2. Set the number of prompts to generate
3. (Optional) Adjust the batch size
4. (Optional) Select models to use from the dropdown
5. (Optional) Configure model selection:
   - Selection Type: 'random' (randomly pick models) or 'sequential' (use models in order)
   - Selection Mode: 'per image' (select for each image) or 'per prompt' (same model for all images of a prompt)
6. (Optional) Configure seed behavior:
   - Check 'Randomize seed for each image' to use different seeds
   - Uncheck to use the same seed for all images
7. Customize the setup prompt, or select "Default" and load it
8. (Optional) Load or save user prompt presets
9. Enter your main prompt describing what you want
10. Click "Generate"

### Model Selection Modes

- **Random Per Image**: Each generated image uses a randomly selected model
- **Sequential Per Image**: Each image uses the next model in the list, cycling through
- **Random Per Prompt**: Each prompt uses a randomly selected model for all its images
- **Sequential Per Prompt**: Each prompt uses the next model in the list for all its images


## Example

System prompt: 
```
Stable Diffusion is a deep learning model that generates images from text prompts. These prompts describe visual elements such as characters, setting, style, lighting, and mood. The syntax supports weighted emphasis using parentheses — for example, (masterpiece:1.5) gives more importance to "masterpiece". Curly brackets can be used to mix concepts, like {blue hair:white hair:0.3}, which blends blue and white hair (30% blue, 70% white).

Example prompt:
masterpiece, (best quality), highly detailed, ultra-detailed, cold, solo, (1girl), (detailed eyes), (shining golden eyes), (long silver hair), expressionless, (long sleeves), (puffy sleeves), (white wings), shining halo, (heavy metal:1.2), (metal jewelry), cross-laced footwear, (chain), (white doves:1.2)


Following the example, provide a set of prompts that detail the following content. Start the prompts directly without using natural language to describe them: 
```

User prompt:
```
Create a cyberpunk street scene
```

Generated output:
```
(1girl), cyberpunk, determined expression, (neon green eyes), (short neon blue hair), (cybernetic implants), (metallic bodysuit:1.2), (glowing circuit patterns), combat boots, (holographic HUD), (tech accessories), standing pose

night scene, neon-lit street, rain-slicked roads, towering skyscrapers, holographic billboards, steam vents, flying vehicles, lens flare, low angle shot, volumetric lighting, dystopian atmosphere, (cyber aesthetics:1.3), dense urban environment

(masterpiece:1.4), (best quality), hyper detailed, ultra realistic, 8k, dramatic lighting, volumetric fog, ray tracing, cinematic composition, professional photography, trending on artstation
```

Note: The output format includes a freeform description followed by three sections of comma-separated tags (subject, environment, quality/style), separated by line breaks. The actual output may vary based on your custom setup prompt.

## File Structure

- `scripts/script.py`: Main extension logic and UI
- `scripts/openai_batcher_config.json`: Persistent config (API key, model, etc.)
- `scripts/system_prompts.json`: System prompt presets
- `scripts/user_prompts.json`: User prompt presets

## License

MIT License - See LICENSE file for details
