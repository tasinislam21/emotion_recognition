import os
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import pandas as pd

# --- CONFIG ---
input_dir = "dataset/images"  # your folder with 100 images
output_dir = "dataset/augmented"
num_augmented = 11000  # target total images
os.makedirs(output_dir, exist_ok=True)
current_label = pd.read_csv('dataset/label.csv')
augment_label = []

# --- Define individual augmentations (subtle) ---
augmentations = [
    transforms.RandomPerspective(distortion_scale=0.3, p=1.0, interpolation=InterpolationMode.BILINEAR),
    transforms.ElasticTransform(alpha=60.0, sigma=3.0, interpolation=InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=1, hue=0.0),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
]

# --- Load original images ---
images = [f for f in os.listdir(input_dir)]

# --- Generate augmented dataset ---
count = 0
for i in tqdm(range(num_augmented)):
    img_name = random.choice(images)
    img_path = os.path.join(input_dir, img_name)
    img = Image.open(img_path).convert("RGB")

    # Pick one random augmentation
    aug = random.choice(augmentations)
    augmented_img = aug(img)

    # Save augmented image
    new_name = f"aug_{i:04d}_{os.path.splitext(img_name)[0]}.jpg"
    augmented_img.save(os.path.join(output_dir, new_name))
    emotion = current_label.loc[current_label['image'] == img_name, 'emotion'].values
    augment_label.append((new_name, emotion[0]))

augmentDF = pd.DataFrame(augment_label, columns=['image', 'emotion'])

augmentDF.to_csv('dataset/aug_label.csv', index=False)
print(f"✅ Generated {count} augmented images in '{output_dir}'")
