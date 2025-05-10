import os
import json
import asyncio
import subprocess
import gradio as gr
from modules import scripts, script_callbacks, shared, sd_models
from modules.processing import process_images
from tqdm import tqdm
import math
import copy
from itertools import chain

# ========== VERSION COMPATIBILITY ==========
try:
    import modules_forge
    forge = True
    ver_bool = True
except ImportError:
    forge = False

if not forge:
    try:
        from packaging import version
        def git_tag():
            try:
                return subprocess.check_output([os.environ.get('GIT', "git"), "describe", "--tags"], shell=False, encoding='utf8').strip()
            except:
                return None
                
        ver = git_tag()
        if not ver:
            try:
                from modules import launch_utils
                ver = launch_utils.git_tag()
            except:
                ver_bool = False
        if ver:
            ver = ver.split('-')[0].rsplit('-', 1)[0]
            ver_bool = version.parse(ver[0:]) >= version.parse("1.7")
    except ImportError:
        print("Python module 'packaging' has not been imported correctly, please try to restart or install it manually.")
        ver_bool = False

# ========== CONFIG SETUP ==========
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "openai_batcher_config.json")


default_setup_prompt = """Stable Diffusion is a deep learning model that generates images from text prompts. These prompts describe visual elements such as characters, setting, style, lighting, and mood. The syntax supports weighted emphasis using parentheses — for example, (masterpiece:1.5) gives more importance to "masterpiece". Curly brackets can be used to mix concepts, like {blue hair:white hair:0.3}, which blends blue and white hair (30% blue, 70% white).

Example prompt:
masterpiece, (best quality), highly detailed, ultra-detailed, cold, solo, (1girl), (detailed eyes), (shining golden eyes), (long silver hair), expressionless, (long sleeves), (puffy sleeves), (white wings), shining halo, (heavy metal:1.2), (metal jewelry), cross-laced footwear, (chain), (white doves:1.2)
Following the example, provide a set of prompts that detail the following content. Start the prompts directly without using natural language to describe them: 
"""

DEFAULT_CONFIG = {
    "api_key": "",
    "api_base": "https://api.openai.com/v1",
    "model": "gpt-3.5-turbo",
    "max_rpm": 60,
    "default_prompt": default_setup_prompt
}


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

# ========== SETTINGS PANEL ==========
def on_ui_settings():
    section = ("openai_batcher", "OpenAI Prompt Generator")
    
    if ver_bool:
        from modules.options import categories
        categories.register_category("openai_batcher", "OpenAI Prompt Generator")
        cat_id = "openai_batcher"
    else:
        cat_id = None

    kwargs = {'category_id': cat_id} if ver_bool else {}

    shared.opts.add_option("openai_api_key",
        shared.OptionInfo("", "OpenAI API Key", section=section, component_args={"type": "password"}, **kwargs))
    shared.opts.add_option("openai_api_base",
        shared.OptionInfo("https://api.openai.com/v1", "OpenAI API Base URL", section=section, **kwargs))
    shared.opts.add_option("openai_model",
        shared.OptionInfo("gpt-3.5-turbo", "OpenAI Model", section=section, **kwargs))
    shared.opts.add_option("openai_max_rpm",
        shared.OptionInfo(60, "Max Requests Per Minute (RPM)", section=section, **kwargs))
    shared.opts.add_option("openai_default_prompt",
        shared.OptionInfo(default_setup_prompt, "Default Setup Prompt", section=section, component_args={"lines": 10, "type": "text"}, **kwargs))

    # Save current settings to config file
    config = {
        "api_key": shared.opts.openai_api_key,
        "api_base": shared.opts.openai_api_base,
        "model": shared.opts.openai_model,
        "max_rpm": int(shared.opts.openai_max_rpm),
        "default_prompt": shared.opts.openai_default_prompt
    }
    save_config(config)

script_callbacks.on_ui_settings(on_ui_settings)

# ========== ASYNC PROMPT BATCHER ==========
from openai import AsyncOpenAI, RateLimitError, APIStatusError
from tqdm.asyncio import tqdm as atqdm

RETRY_LIMIT = 5
RETRY_BACKOFF = 2

async def async_gen_prompt(setup_prompt, main_prompt, api_config, session_id=""):
    messages = [
        {"role": "system", "content": setup_prompt},
        {"role": "user", "content": main_prompt},
    ]

    api_key = api_config.get("api_key", "")
    api_base = api_config.get("api_base", "https://api.openai.com/v1")
    model = api_config.get("model", "gpt-3.5-turbo")

    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    for attempt in range(RETRY_LIMIT):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=60,
            )

            headers = getattr(response, "response_headers", {})
            remaining = headers.get("x-ratelimit-remaining-requests")
            reset = headers.get("x-ratelimit-reset-requests")

            if remaining and reset and int(remaining) < 2:
                wait_time = int(reset)
                print(f"[{session_id}] Near rate limit, sleeping {wait_time}s...")
                await asyncio.sleep(wait_time)

            return response.choices[0].message.content

        except (RateLimitError, APIStatusError) as e:
            wait = RETRY_BACKOFF ** attempt
            print(f"[{session_id}] Retry {attempt + 1}/{RETRY_LIMIT} in {wait}s due to {e}")
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"[{session_id}] Unhandled error: {e}")
            raise

    raise Exception(f"[{session_id}] Failed after {RETRY_LIMIT} retries.")

async def batch_generate(setup_prompt, main_prompt, count, api_config):
    tasks = [
        async_gen_prompt(setup_prompt, main_prompt, api_config, session_id=f"Prompt-{i+1}")
        for i in range(count)
    ]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=count, desc="Generating prompts"):
        result = await coro
        shared.state.job_no += 1
        results.append(result)
    return results

# ========== PROMPT PRESET HELPERS ==========
SYSTEM_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "system_prompts.json")
USER_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "user_prompts.json")

def normalize_title(title):
    if title is None:
        return ''
    return ''.join(c for c in title.lower() if c.isalnum())

def load_prompts(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_prompts(path, prompts):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2)

def get_prompt_titles(prompts):
    return list(prompts.keys())

def get_prompt_by_title(prompts, title):
    norm = normalize_title(title)
    for t, p in prompts.items():
        if normalize_title(t) == norm:
            return p
    return None

def set_prompt(prompts, title, value):
    # Overwrite or add new
    prompts[title] = value
    return prompts

# ========== UI HELPERS ==========
def build_prompt_preset_row(label, textbox, preset_type):
    # preset_type: 'system' or 'user'
    path = SYSTEM_PROMPTS_PATH if preset_type == 'system' else USER_PROMPTS_PATH
    prompts = load_prompts(path)
    titles = get_prompt_titles(prompts)
    if preset_type == 'system':
        titles = ['Default'] + [t for t in titles if normalize_title(t) != 'default']
    with gr.Row():
        preset_dropdown = gr.Dropdown(label=f"{label} Presets", choices=titles, value=None, interactive=True)
        load_btn = gr.Button("Load")
        # Save button is disabled for 'Default' in system presets
        save_btn = gr.Button("Save", interactive=False if preset_type == 'system' else True)
    # Save dialog (hidden by default)
    with gr.Row(visible=False) as save_row:
        save_title = gr.Dropdown(label="Title", choices=titles, value=None, allow_custom_value=True)
        save_confirm = gr.Button("Confirm Save")
        save_cancel = gr.Button("Cancel")
    return preset_dropdown, load_btn, save_btn, save_row, save_title, save_confirm, save_cancel

# ========== MAIN SCRIPT CLASS ==========
class Script(scripts.Script):
    def title(self):
        return "GPT Prompts"

    def ui(self, _):
        def get_default_prompt():
            return shared.opts.openai_default_prompt or default_setup_prompt

        with gr.Row():
            prompts_count = gr.Slider(label='Prompts Count', value=1, minimum=1, maximum=30, step=1)
        with gr.Row():
            batch_size = gr.Slider(label='Batch Size', value=1, minimum=1, maximum=100, step=1)

        with gr.Row():
            model_choices = gr.Dropdown(label='Models to use', choices=sorted(sd_models.checkpoints_list.keys(), key=str.casefold), multiselect=True, elem_id=self.elem_id("model_choices"))

        with gr.Row():
            with gr.Column():
                selection_type = gr.Dropdown(label='Selection Type', choices=['random', 'sequential'], value='random', elem_id=self.elem_id("selection_type"))
            with gr.Column():
                selection_mode = gr.Dropdown(label='Selection Mode', choices=['per image', 'per prompt'], value='per image', elem_id=self.elem_id("selection_mode"))
            with gr.Column():
                randomize_seed = gr.Checkbox(label='Randomize seed for each image', value=True, elem_id=self.elem_id("randomize_seed"))
                randomize_orientation = gr.Checkbox(label='Randomly flip orientation', value=False, elem_id=self.elem_id("randomize_orientation"))

        # Setup Prompt UI
        with gr.Row():
            setup_prompt = gr.Textbox(
                label='Setup Prompt',
                value=get_default_prompt(),
                lines=5,
                elem_id=self.elem_id("gpt_script_setup_prompt")
            )
        # Preset row for Setup Prompt
        setup_preset_dropdown, setup_load_btn, setup_save_btn, setup_save_row, setup_save_title, setup_save_confirm, setup_save_cancel = build_prompt_preset_row("System", setup_prompt, 'system')

        # User Prompt UI
        with gr.Row():
            main_prompt = gr.Textbox(label='User Prompt', lines=10)
        # Preset row for User Prompt
        user_preset_dropdown, user_load_btn, user_save_btn, user_save_row, user_save_title, user_save_confirm, user_save_cancel = build_prompt_preset_row("User", main_prompt, 'user')

        # ========== EVENT LOGIC FOR PRESET UI ========== #
        # Helper to wire up a preset row (system/user)
        def wire_preset_row(preset_dropdown, load_btn, save_btn, save_row, save_title, save_confirm, save_cancel, textbox, preset_type):
            path = SYSTEM_PROMPTS_PATH if preset_type == 'system' else USER_PROMPTS_PATH

            def on_dropdown_change(selected_title):
                prompts = load_prompts(path)
                if preset_type == 'system' and selected_title == 'Default':
                    # Only update button states, do not update textbox
                    return gr.update(interactive=True), gr.update(interactive=False), gr.update()
                # Only update button states, do not update textbox
                return gr.update(interactive=True), gr.update(interactive=True), gr.update()

            def on_load_click(selected_title):
                if preset_type == 'system' and selected_title == 'Default':
                    return shared.opts.openai_default_prompt or default_setup_prompt
                prompts = load_prompts(path)
                preset = get_prompt_by_title(prompts, selected_title) or ""
                return preset

            def on_save_click():
                prompts = load_prompts(path)
                titles = get_prompt_titles(prompts)
                return gr.update(visible=True), gr.update(choices=titles, value=None)

            def on_save_confirm(title, value):
                prompts = load_prompts(path)
                prompts = set_prompt(prompts, title, value)
                save_prompts(path, prompts)
                titles = get_prompt_titles(prompts)
                return (
                    gr.update(visible=False),  # Hide save row
                    gr.update(choices=titles, value=title),  # Update dropdown
                    gr.update(choices=titles, value=title),  # Update save title dropdown
                )

            def on_save_cancel():
                return gr.update(visible=False)

            # Textbox change event: unselect preset and enable Save, but do NOT clear textbox
            def on_textbox_change(new_value, event=None):
                # Only unselect dropdown if the change was from user input, not programmatic (e.g., Load)
                if event and hasattr(event, 'source') and event.source == 'user':
                    return None, gr.update(interactive=True)
                return gr.update(), gr.update(interactive=True)

            # Wire up events
            preset_dropdown.change(
                fn=on_dropdown_change,
                inputs=[preset_dropdown],
                outputs=[load_btn, save_btn, textbox],
            )
            load_btn.click(
                fn=on_load_click,
                inputs=[preset_dropdown],
                outputs=[textbox],  # Only update textbox, not dropdown
            )
            save_btn.click(
                fn=on_save_click,
                inputs=None,
                outputs=[save_row, save_title],
            )
            save_confirm.click(
                fn=on_save_confirm,
                inputs=[save_title, textbox],
                outputs=[save_row, preset_dropdown, save_title],
            )
            save_cancel.click(
                fn=on_save_cancel,
                inputs=None,
                outputs=[save_row],
            )
            # Restore: Textbox change event
            textbox.change(
                fn=on_textbox_change,
                inputs=[textbox],
                outputs=[preset_dropdown, save_btn],
                show_progress=False,
            )

        # Wire up both preset rows
        wire_preset_row(setup_preset_dropdown, setup_load_btn, setup_save_btn, setup_save_row, setup_save_title, setup_save_confirm, setup_save_cancel, setup_prompt, 'system')
        wire_preset_row(user_preset_dropdown, user_load_btn, user_save_btn, user_save_row, user_save_title, user_save_confirm, user_save_cancel, main_prompt, 'user')

        return [setup_prompt, main_prompt, prompts_count, batch_size, model_choices, selection_type, selection_mode, randomize_seed, randomize_orientation]

    def run(self, p, setup_prompt, main_prompt, prompts_count, batch_size, model_choices, selection_type, selection_mode, randomize_seed, randomize_orientation):
        from modules.processing import fix_seed, Processed, process_images
        import random
        
        original_seed = p.seed
        if not randomize_seed:
            fix_seed(p)

        api_config = {
            "api_key": shared.opts.openai_api_key or load_config().get("api_key"),
            "api_base": shared.opts.openai_api_base or load_config().get("api_base"),
            "model": shared.opts.openai_model or load_config().get("model"),
            "max_rpm": shared.opts.openai_max_rpm or load_config().get("max_rpm", 60),
        }

        print(f"🔁 Generating {prompts_count} prompt(s)...")
        shared.state.textinfo = f"Generating {prompts_count} prompts..."
        shared.state.job_count = prompts_count
        shared.state.job_no = 0

        raw_prompts = asyncio.run(batch_generate(setup_prompt, main_prompt, prompts_count, api_config))

        shared.state.textinfo = f"Generating {prompts_count} prompts... Done"
        shared.state.job_no = prompts_count
        print(f"📸 Total prompts: {len(raw_prompts)}")

        processed_results = []
        total_images = len(raw_prompts) * batch_size
        current_model_index = 0

        # Pre-select models for 'per prompt' mode
        prompt_models = {}
        if selection_mode == 'per prompt' and model_choices:
            selected_models = [sd_models.get_closet_checkpoint_match(title).name for title in model_choices]
            if selected_models:
                for i, _ in enumerate(raw_prompts):
                    if selection_type == 'sequential':
                        prompt_models[i] = selected_models[current_model_index % len(selected_models)]
                        current_model_index += 1
                    else:  # random
                        prompt_models[i] = random.choice(selected_models)

        shared.state.textinfo = f"Generating {total_images} images..."
        shared.state.job_count = total_images
        shared.state.job_no = 0

        # Process each image individually
        image_no = 0
        for prompt_idx, prompt in enumerate(raw_prompts):
            for batch_idx in range(batch_size):
                shared.state.job_no = image_no
                shared.state.textinfo = f"Generating... Prompt {prompt_idx+1} of {prompts_count}. Image {batch_idx+1} of {batch_size}. Total: {image_no+1} of {total_images}."
                p_copy = copy.copy(p)
                p_copy.prompt = prompt
                p_copy.n_iter = 1  # Process one image at a time
                
                # Handle seed
                if randomize_seed:
                    p_copy.seed = random.randint(0, 2**32 - 1)
                    print(f"🎲 Using seed: {p_copy.seed}")
                else:
                    p_copy.seed = original_seed
                
                # Handle orientation
                if randomize_orientation and random.random() < 0.5:
                    p_copy.width, p_copy.height = p_copy.height, p_copy.width
                    print(f"🔄 Flipped orientation to {p_copy.width}x{p_copy.height}")
                
                # Handle model selection
                if model_choices:
                    selected_models = [sd_models.get_closet_checkpoint_match(title).name for title in model_choices]
                    if selected_models:
                        if selection_mode == 'per prompt':
                            model = prompt_models[prompt_idx]
                        else:  # per image
                            if selection_type == 'sequential':
                                model = selected_models[current_model_index % len(selected_models)]
                                current_model_index += 1
                            else:  # random
                                model = random.choice(selected_models)
                        
                        print(f"🎨 Using model: {model}")
                        p_copy.override_settings = {"sd_model_checkpoint": model}
                        p_copy.override_settings_restore_afterwards = True
                
                # Process single image
                image_no += 1
                processed = process_images(p_copy)
                processed_results.append(processed)
                shared.state.textinfo = f"Processed {shared.state.job_no}/{total_images} images"
        
        # Combine all results
        if len(processed_results) == 0:
            return Processed(p, [], p.seed, "")

        combined = copy.copy(processed_results[0])
        combined.images = []
        combined.all_prompts = []
        combined.all_seeds = []
        combined.infotexts = []
        
        for processed in processed_results:
            if processed.images:
                combined.images.extend(processed.images)
                combined.all_prompts.extend(processed.all_prompts)
                combined.all_seeds.extend(processed.all_seeds)
                combined.infotexts.extend(processed.infotexts)

        if len(combined.images) == 0:
            print("No images were generated!")
            return Processed(p, [], p.seed, "")

        return combined
