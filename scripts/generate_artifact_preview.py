from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

sample_path = Path('data_5_fold/fold_1/test/subluxation/test_subluxation_0664_f6ad98f2fa.jpg')
out_dir = Path('chapter4_results/sample_artifacts_preview')
out_dir.mkdir(parents=True, exist_ok=True)


def ensure_uint8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def motion_blur_pil(img, kernel_size=9):
    arr = np.asarray(img).astype(np.float32)
    pad = kernel_size // 2
    padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode='edge')
    out = np.zeros_like(arr)
    for k in range(kernel_size):
        out += padded[:, k:k + arr.shape[1], :]
    out /= kernel_size
    return Image.fromarray(ensure_uint8(out))


def gaussian_noise_pil(img, sigma=18.0):
    rng = np.random.default_rng(42)
    arr = np.asarray(img).astype(np.float32)
    return Image.fromarray(ensure_uint8(arr + rng.normal(0, sigma, arr.shape)))


def metal_streak_pil(img, num_streaks=3):
    rng = np.random.default_rng(42)
    base = img.convert('RGB')
    overlay = Image.new('RGB', base.size, (0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size
    for _ in range(num_streaks):
        intensity = int(rng.integers(180, 255))
        width = int(rng.integers(2, 8))
        draw.line(
            (
                int(rng.integers(0, w)),
                int(rng.integers(0, h)),
                int(rng.integers(0, w)),
                int(rng.integers(0, h)),
            ),
            fill=(intensity, intensity, intensity),
            width=width,
        )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2.0))
    out = np.maximum(np.asarray(base), np.asarray(overlay))
    return Image.fromarray(out.astype(np.uint8))


img = Image.open(sample_path).convert('RGB').resize((224, 224))
versions = {
    'none_clean': img,
    'motion_blur': motion_blur_pil(img),
    'gaussian_noise': gaussian_noise_pil(img),
    'metal_streak': metal_streak_pil(img),
}

for name, im in versions.items():
    im.save(out_dir / f'{name}.png')

montage = Image.new('RGB', (224 * 4, 224 + 44), 'white')
draw = ImageDraw.Draw(montage)
for i, (name, im) in enumerate(versions.items()):
    x = i * 224
    montage.paste(im, (x, 30))
    draw.text((x + 5, 8), name.replace('_', ' '), fill='black')
montage.save(out_dir / 'artifact_preview_montage.png')

print('sample_path=' + str(sample_path))
print('output_dir=' + str(out_dir))
print('montage=' + str(out_dir / 'artifact_preview_montage.png'))
