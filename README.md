# AetherSketch: Custom Stable Diffusion LoRA Training on Google Colab

This repository contains the code and a detailed guide for fine-tuning a Stable Diffusion model using **LoRA (Low-Rank Adaptation)** to generate images in a specific artistic style, such as a modern anime or manhwa. The entire pipeline is designed to be run in a **Google Colab** environment, leveraging its free GPU resources.

---

### Project Goal and Methodology

The primary objective of this project is to create a functional and repeatable pipeline for fine-tuning a Stable Diffusion model. The methodology involves using open-source tools from the Hugging Face ecosystem, including the `diffusers`, `transformers`, and `datasets` libraries.

---

### Demo

Here is a quick demonstration of the model in action, showcasing its ability to generate images in the target art style.

![AetherSketch Demo GIF](demo.gif)

---

### 1. Initial Setup and Configuration

This first code block installs all the necessary libraries and dependencies, mounts your Google Drive for persistent storage, and clones the `diffusers` repository from Hugging Face to get the training scripts. It is a good practice to run this at the beginning of every session.

```python
# Install necessary libraries
!pip install -qq accelerate diffusers transformers datasets
!pip install -qq torch --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
!pip install -qq peft
!pip install -qq requests

# Uninstall the old diffusers version if it exists and install the latest version from source to avoid version mismatch errors
!pip uninstall -y diffusers
!pip install -qq git+[https://github.com/huggingface/diffusers.git](https://github.com/huggingface/diffusers.git)

# Mount Google Drive to save the trained model weights. This is crucial for persistent storage.
from google.colab import drive
drive.mount('/content/gdrive')

# Set up your Hugging Face API token for model and dataset access.
# Make sure to replace "hf_xxxxxxxxxxxxxxxxxxxxxxxxxx" with your actual token.
import os
os.environ["HF_TOKEN"] = "HF_TOKEN"

# Clone the diffusers repository to access the training scripts
!git clone [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)

# Navigate to the correct directory containing the training script
os.chdir('/content/diffusers/examples/text_to_image')
```
### 2. Dataset Preparation
Before training, you need to prepare your dataset. The placeholder code uses a public dataset, but you should replace it with your own high-quality, captioned dataset of images in the desired art style.

``` Python

from datasets import load_dataset

# Define the output directory on your Google Drive to store the LoRA model
output_dir = "/content/gdrive/MyDrive/manga_manhwa_lora_model"

# Create the output directory if it doesn't exist
!mkdir -p "$output_dir"

# NOTE: The following line uses a placeholder dataset.
# For your project, you must replace "linoyts/rubber_ducks" with your actual dataset.
# The dataset should contain images of your desired manga/manhwa style with descriptive captions.
dataset_id = "linoyts/rubber_ducks"
dataset = load_dataset(dataset_id, split="train")

print(f"Dataset loaded with {len(dataset)} examples.")
```
### 3. LoRA Model Training
This is the core training command. It uses accelerate launch to start the fine-tuning process. You can modify the parameters to suit your needs, such as the learning rate, number of epochs, and resolution.

``` Python

!accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --dataset_name="linoyts/rubber_ducks" \
    --output_dir="$output_dir" \
    --caption_column="prompt" \
    --resolution=512 \
    --train_batch_size=4 \
    --num_train_epochs=10 \
    --checkpointing_steps=500 \
    --learning_rate=1e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=50 \
    --seed=42 \
    --mixed_precision="fp16"
```
### 4. Inference (Image Generation)
After training is complete, you can use the following code to load your trained LoRA weights and generate new images. This script has been updated to accept prompts via keyboard input, allowing for interactive image generation.

``` Python

from diffusers import StableDiffusionPipeline
import torch
import os

# Navigate back to the main content directory
os.chdir('/content/')

# Load the base pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# Load the trained LoRA weights
lora_model_path = "/content/gdrive/MyDrive/manga_manhwa_lora_model/pytorch_lora_weights.safetensors"
pipeline.unet.load_attn_procs(lora_model_path)

# Move pipeline to GPU
pipeline.to("cuda")

# Use a loop to continuously ask for new prompts
while True:
    try:
        # Take a prompt as keyboard input from the user
        prompt = input("Enter your prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            print("Exiting image generation.")
            break

        # Run inference with the user-provided prompt
        with torch.autocast("cuda"):
            image = pipeline(prompt).images[0]

        # Define a path to save your generated image on Google Drive
        generated_image_path = "/content/gdrive/MyDrive/manga_manhwa_lora_model/generated_manga_image.png"
        image.save(generated_image_path)
        print(f"Generated image saved to {generated_image_path}")

        # Display the generated image
        from IPython.display import display
        display(image)

    except Exception as e:
        print(f"An error occurred: {e}")
```
### 5. Troubleshooting and Problem Resolution
-- During the project, several common issues were encountered and resolved. This section summarizes those problems and their solutions for future reference.

-- Command Syntax and Path Errors: Initial attempts to run the training script failed due to an incorrect working directory. This was resolved by separating the Python os.chdir() command into its own cell and correcting the path from /examples/lora to the current /examples/text_to_image.

-- Version Mismatch (ImportError): The script required a newer version of the diffusers library. The fix was to uninstall the old version and install the latest one directly from the Hugging Face GitHub source using pip install git+https://github.com/huggingface/diffusers.git.

-- Incorrect Dataset Column (ValueError): The accelerate launch command failed because the --caption_column argument had an incorrect value. This was fixed by inspecting the dataset and updating the argument to "prompt", which was the correct column name for the image captions.

