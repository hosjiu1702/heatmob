"""
This script is just a convenient way to use Huggingface's Accelerate to train your
txt2img model with LoRA
"""
import subprocess


training_script = 'train_text_to_image_lora.py'
base_model = 'train_text_to_image_lora.py'
learning_rate = 1e-4
output_dir = '/models'

def train():
    subprocess.run(['accelerate',f'{training_script}',
                   f'{}'])


if __name__ == '__main__':
    train()