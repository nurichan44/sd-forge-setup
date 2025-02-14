import os
import subprocess
import json
import requests
from urllib.parse import urlparse

# --- Constants (Do not change) ---
FORGE_REPO = "https://github.com/lllyasviel/stable-diffusion-webui-forge"
CONFIG_FILE = "config.json"

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "output_path": "AI_PICS",
    "username": "a",
    "password": "a",
    "ngrok_token": "",
    "forge_version": "",  # Leave blank for latest
    "use_google_drive": False, # Google Drive option removed
    "models": {
        "Flux1_dev": True,
        "Flux1_schnell": True,
        "SDXL_1": False,
        "JuggernautXL_v8": False,
        "Pony_Diffusion_XL_v6": False,
        "v1_5_model": False,
        "F222_model": False,
        "Realistic_Vision_model": False,
        "Realistic_Vision_Inpainting_model": False,
        "DreamShaper_model": False,
        "DreamShaper_Inpainting_model": False,
        "OpenJourney_model": False,
        "Anything_v3_model": False,
        "Inkpunk_Diffusion_model": False,
        "SD_1_5_ControlNet_models": False,
        "SDXL_ControlNet_models": False,
        "IP_Adapter_models": False,
    },
    "extensions": {
        "Aspect_Ratio_Helper": True,
        "Infinite_Image_Browser": True,
    },
    "extra_models": {
        "Checkpoint_models_from_URL": "",
        "LoRA_models_from_URL": "",
    },
    "Civitai_API_Key": "",
    "extra_extensions": {
        "Extensions_from_URL": "",
    },
    "extra_args": "",
    "clear_output": True,
}

# --- Whitelist of main models (don't redownload if config is false but file exists) ---
MAIN_MODELS_WHITELIST = [
    "Flux1_dev",
    "Flux1_schnell",
    "SDXL_1",
    "JuggernautXL_v8",
    "Pony_Diffusion_XL_v6",
    "v1_5_model",
    "F222_model",
    "Realistic_Vision_model",
    "Realistic_Vision_Inpainting_model",
    "DreamShaper_model",
    "DreamShaper_Inpainting_model",
    "OpenJourney_model",
    "Anything_v3_model",
    "Inkpunk_Diffusion_model",
]

def load_config():
    """Loads configuration from config.json, or creates it with defaults."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            try:
                config = json.load(f)
                # Merge with defaults to add any new options
                config = {**DEFAULT_CONFIG, **config}
                # Ensure nested dictionaries are also merged
                for key in DEFAULT_CONFIG:
                    if isinstance(DEFAULT_CONFIG[key], dict) and key in config:
                        config[key] = {**DEFAULT_CONFIG[key], **config[key]}
                return config
            except json.JSONDecodeError:
                print("Error: config.json is corrupted. Using default settings.")
                return DEFAULT_CONFIG
    else:
        return DEFAULT_CONFIG


def save_config(config):
    """Saves the configuration to config.json."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def download_file(url, destination_folder, filename=None):
    """Downloads a file, handling Hugging Face and Civitai URLs."""
    parsed_url = urlparse(url)
    if not filename:
        filename = os.path.basename(parsed_url.path)

    destination_path = os.path.join(destination_folder, filename)

    if parsed_url.netloc == "huggingface.co":
        print(f"Downloading from Hugging Face: {url} to {destination_path}")
        command = ["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", url, "-o", filename]
        subprocess.run(command, cwd=destination_folder, check=True)

    elif parsed_url.netloc == "civitai.com":
        print(f"Downloading from Civitai: {url} to {destination_path}")
        if config["Civitai_API_Key"]:
            url += f"&token={config['Civitai_API_Key']}" if "?" in url else f"?token={config['Civitai_API_Key']}"
        command = ["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", url, "-o", filename]
        subprocess.run(command, cwd=destination_folder, check=True)

    else:  # Direct download with requests
        print(f"Downloading from {url} to {destination_path}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(destination_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            exit(1)

def get_model_filename(model_name):
    """Returns the expected filename for a given model name."""
    # Add more mappings as needed
    if model_name == "Flux1_dev": return "flux1-dev-bnb-nf4-v2.safetensors"
    if model_name == "Flux1_schnell": return "flux1-schnell-bnb-nf4.safetensors"
    if model_name == "SDXL_1": return "sd_xl_base_1.0.safetensors"  # Only base for SDXL_1
    if model_name == "JuggernautXL_v8": return "juggernautXL_v8RundiffusionSDXL_v80.safetensors"
    if model_name == "Pony_Diffusion_XL_v6": return "ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
    if model_name == "v1_5_model": return "v1-5-pruned-emaonly.safetensors"
    if model_name == "F222_model": return "f222.ckpt"
    if model_name == "Realistic_Vision_model": return "Realistic_Vision_V5.1_fp16-no-ema.safetensors"
    if model_name == "Realistic_Vision_Inpainting_model": return "Realistic_Vision_V5.1-inpainting.safetensors"
    if model_name == "DreamShaper_model": return "DreamShaper_8_pruned.safetensors"
    if model_name == "DreamShaper_Inpainting_model": return "DreamShaper_8_INPAINTING.inpainting.safetensors"
    if model_name == "OpenJourney_model": return "mdjrny-v4.ckpt"
    if model_name == "Anything_v3_model": return "Anything-V3.0-pruned-fp16.safetensors"
    if model_name == "Inkpunk_Diffusion_model": return "Inkpunk-Diffusion-v2.ckpt"

    return None  # Unknown model

def download_models(root, config):
    """Downloads selected models."""
    models_dir = os.path.join(root, "models", "Stable-diffusion")
    os.makedirs(models_dir, exist_ok=True)
    print('⏳ Downloading models ...')

    for model_name, download in config["models"].items():
        filename = get_model_filename(model_name)
        if not filename:  # Skip unknown models
            continue

        filepath = os.path.join(models_dir, filename)
        should_download = False

        if model_name in MAIN_MODELS_WHITELIST:
            if download or not os.path.exists(filepath):
                # Download if config is true OR file doesn't exist
                should_download = True
        else:  # Other models (ControlNet, LoRA, etc.)
            if download and not os.path.exists(filepath):
                should_download = True

        if should_download:
            if model_name == "Flux1_dev":
                download_file('https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4-v2.safetensors', models_dir)
            elif model_name == "Flux1_schnell":
                download_file('https://huggingface.co/silveroxides/flux1-nf4-weights/resolve/main/flux1-schnell-bnb-nf4.safetensors', models_dir)
            elif model_name == "SDXL_1":
                download_file('https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors', models_dir)
                # Download refiner ONLY if base is downloaded
                download_file('https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors', models_dir)
            elif model_name == "JuggernautXL_v8":
                download_file('https://civitai.com/api/download/models/288982', models_dir)
            elif model_name == "Pony_Diffusion_XL_v6":
                download_file('https://huggingface.co/Magamanny/Pony-Diffusion-V6-XL/resolve/main/ponyDiffusionV6XL_v6StartWithThisOne.safetensors', models_dir)
            elif model_name == "v1_5_model":
                download_file('https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors', models_dir)
            elif model_name == "F222_model":
                download_file('https://huggingface.co/acheong08/f222/resolve/main/f222.ckpt', models_dir)
            elif model_name == "Realistic_Vision_model":
                download_file('https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_fp16-no-ema.safetensors', models_dir)
            elif model_name == "Realistic_Vision_Inpainting_model":
                download_file('https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1-inpainting.safetensors', models_dir)
            elif model_name == "DreamShaper_model":
                download_file('https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors', models_dir)
            elif model_name == "DreamShaper_Inpainting_model":
                download_file('https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_INPAINTING.inpainting.safetensors', models_dir)
            elif model_name == "OpenJourney_model":
                download_file('https://huggingface.co/prompthero/openjourney/resolve/main/mdjrny-v4.ckpt', models_dir)
            elif model_name == "Anything_v3_model":
                download_file('https://huggingface.co/admruul/anything-v3.0/resolve/main/Anything-V3.0-pruned-fp16.safetensors', models_dir)
            elif model_name == "Inkpunk_Diffusion_model":
                download_file('https://huggingface.co/Envvi/Inkpunk-Diffusion/resolve/main/Inkpunk-Diffusion-v2.ckpt', models_dir)

    if config["extra_models"]["Checkpoint_models_from_URL"]:
        for model_url in config["extra_models"]["Checkpoint_models_from_URL"].split(','):
            download_file(model_url.strip(), models_dir)

    if config["extra_models"]["LoRA_models_from_URL"]:
        lora_dir = os.path.join(root, "models", "Lora")
        os.makedirs(lora_dir, exist_ok=True)
        for model_url in config["extra_models"]["LoRA_models_from_URL"].split(','):
            download_file(model_url.strip(), lora_dir)

    # Download VAEs
    vae_dir = os.path.join(root, "models", "VAE")
    os.makedirs(vae_dir, exist_ok=True)
    download_file('https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.ckpt', vae_dir, filename='vae-ft-ema-560000-ema-pruned.ckpt')
    download_file('https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt', vae_dir, filename='vae-ft-mse-840000-ema-pruned.ckpt')

def install_controlnet_models(root, config):
    """Downloads ControlNet models."""
    controlnet_dir = os.path.join(root, "models", "ControlNet")
    os.makedirs(controlnet_dir, exist_ok=True)

    if config["models"]["SD_1_5_ControlNet_models"]:
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime.pth', controlnet_dir)
        download_file('https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_color_sd14v1.pth', controlnet_dir)
        download_file('https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_style_sd14v1.pth', controlnet_dir)
        download_file('https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/control_v1p_sd15_qrcode_monster.safetensors', controlnet_dir)
        download_file('https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/v2/control_v1p_sd15_qrcode_monster_v2.safetensors', controlnet_dir)

    if config["models"]["SDXL_ControlNet_models"]:
        download_file('https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors', controlnet_dir, filename='diffusion_xl_openpose.safetensors')
        download_file('https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_full.safetensors', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_depth_mid.safetensors', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ip-adapter_xl.pth', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur.safetensors', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur_anime.safetensors', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_scribble_anime.safetensors', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_recolor_256lora.safetensors', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_sketch_256lora.safetensors', controlnet_dir)
        download_file('https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_lineart.safetensors', controlnet_dir)

    if config["models"]["IP_Adapter_models"]:
        download_file('https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors', controlnet_dir)
        download_file('https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors', controlnet_dir)
        download_file('https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors', controlnet_dir)
        download_file('https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin', controlnet_dir)
        download_file('https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors', controlnet_dir)
        download_file('https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin', controlnet_dir)
        try:
            subprocess.run(["pip", "install", "insightface"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error installing insightface: {e}")
            exit(1)


def install_extensions_from_url(urls, extensions_dir):
    """Installs extensions from given URLs."""
    for url in urls.split(','):
        url = url.strip()
        if not url:  # Skip empty URLs
            continue
        print(f"Cloning extension from URL: {url}")
        try:
            subprocess.run(["git", "clone", url], cwd=extensions_dir, check=True)
            repo_name = os.path.basename(urlparse(url).path)
            repo_path = os.path.join(extensions_dir, repo_name)
            #Remove .git
            git_dir = os.path.join(repo_path, ".git")
            if os.path.exists(git_dir):
                subprocess.run(["rm", "-rf", git_dir], check=True)

        except subprocess.CalledProcessError as e:
            print(f"Error cloning {url}: {e}")
            # Don't exit, try to install other extensions
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def install_infinite_image_browser(extensions_dir):
    """Installs and configures the Infinite Image Browser extension."""
    install_extensions_from_url('https://github.com/zanllp/sd-webui-infinite-image-browsing', extensions_dir)
    iib_dir = os.path.join(extensions_dir, "sd-webui-infinite-image-browsing")
    env_file = os.path.join(iib_dir, ".env")
    # Create .env file from .env.example, setting IIB_SECRET_KEY
    try:
        with open(os.path.join(iib_dir, ".env.example"), "r") as example_file:
            content = example_file.read()
            content = content.replace("IIB_SECRET_KEY=", "IIB_SECRET_KEY=SDA")
        with open(env_file, "w") as env_file_out:
            env_file_out.write(content)
    except FileNotFoundError:
        print("Error: .env.example not found in sd-webui-infinite-image-browsing.")
        exit(1)
    except Exception as e:
        print(f"Error configuring Infinite Image Browser: {e}")
        exit(1)

def prepare_launch_args(config):
    """Prepares the arguments for launch.py based on the configuration."""
    args = [
        "python",
        "launch.py",
        "--gradio-img2img-tool", "color-sketch",
        "--enable-insecure-extension-access",
        "--gradio-queue",
    ]
    if config["ngrok_token"]:
        args.extend(["--ngrok", config["ngrok_token"]])
    else:
        args.append("--share")

    if config["username"] and config["password"]:
        args.extend(["--gradio-auth", f"{config['username']}:{config['password']}"])

    args.extend(config["extra_args"].split())  # Add extra arguments
    return args

def main():
    """Main function to install or update Stable Diffusion Forge."""
    global config  # Use the global config variable
    config = load_config()  # Load configuration
    root = os.getcwd()  # Get current working directory
    forge_path = os.path.join(root, "webui")
    extensions_dir = os.path.join(forge_path, "extensions")

    # --- 1. Clone or update Forge ---
    if not os.path.exists(forge_path):
        print("Cloning Forge repository...")
        try:
            subprocess.run(["git", "clone", FORGE_REPO, forge_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error cloning Forge repository: {e}")
            exit(1)
    else:
        print("Forge repository already exists. Updating...")
        try:
            subprocess.run(["git", "pull"], cwd=forge_path, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error updating Forge repository: {e}")
            exit(1)

    if config['forge_version']:
      try:
        print(f"Checking out version: {config['forge_version']}")
        subprocess.run(["git", "checkout", "-f", config['forge_version']], cwd=forge_path, check=True)
      except subprocess.CalledProcessError as e:
        print(f"Error checking out version {config['forge_version']}: {e}")
        exit(1)

    # --- 2. First launch (to install dependencies) ONLY ON FIRST INSTALL ---
    if not os.path.exists(os.path.join(forge_path, "venv")):
        print("Performing first launch to install dependencies...")
        try:
            subprocess.run(["python", "launch.py", "--exit"], cwd=forge_path, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during first launch: {e}")
            exit(1)


    # --- 3. Install extensions ---
    os.makedirs(extensions_dir, exist_ok=True)

    if config["extensions"]["Aspect_Ratio_Helper"]:
        install_extensions_from_url('https://github.com/altoiddealer/--sd-webui-ar-plusplus', extensions_dir)
    if config["extensions"]["Infinite_Image_Browser"]:
        install_infinite_image_browser(extensions_dir)
    if config["extra_extensions"]["Extensions_from_URL"]:
        install_extensions_from_url(config["extra_extensions"]["Extensions_from_URL"], extensions_dir)


    # --- 4. Downgrade httpx (if necessary) ---
    try:
        subprocess.run(["pip", "install", "httpx==0.24.1"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downgrading httpx: {e}")
        exit(1)


    # --- 5. Download models ---
    download_models(forge_path, config)  # Pass forge_path
    install_controlnet_models(forge_path, config) # Pass forge_path

    # --- 6. Prepare launch arguments ---
    launch_args = prepare_launch_args(config)

    # --- 7. Save configuration ---
    save_config(config)

    # --- 8. Clear output (optional) ---
    if config["clear_output"]:
       os.system('cls' if os.name == 'nt' else 'clear')

	# --- 9. Launch Forge ---
    # We don't launch here anymore.  The .bat file handles that.
    # subprocess.run(launch_args, cwd=forge_path, check=False)
    print("Setup/update complete.")


if __name__ == "__main__":
    # Install aria2 (if not already installed) - pyngrok is no longer needed
    try:
        subprocess.run(["pip", "install", "aria2"], check=True)
    except subprocess.CalledProcessError as e:
         print(f"Error installing aria2: {e}")
         exit(1)
    main()
