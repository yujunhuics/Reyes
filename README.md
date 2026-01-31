# Reyes（睿视）



# 2026.01

- 权重开源：https://modelscope.cn/models/yujunhuinlp/Reyes-0.6B



## 模型架构

得益于开源社区优秀的开源模型（qwenvl、smolvlm等）在模型、代码、训练等提供的思路，Reyes-0.6B整体结构遵循经典的Vit+两层MLP+LLM架构：
- vit视觉编码器：SigLIP2-Base-Patch16-512
- LLM：qwen3-0.6B


#### 优化trick

##### 原生分辨率支持
在上个版本Reyes-8B中，主要采用了动态分辨率对图像进行预处理，包括归一化、缩放、裁剪、根据宽高比动态处理等操作。


在《[多模态大模型中不同分辨率策略研究与原生分辨率的有效性评估](https://mp.weixin.qq.com/s/JoxZPH9q-kBRj4Ecaj1KIQ)》和现有多个VLMs（如qwenvl、keye-vl等）中都使用了原生分辨率。


因此本次Reyes-0.6B模型也增加了原生分辨率的支持，通过适配集成 2D Rotary Position Embeddings（2D-RoPE）和双三次插值适配位置嵌入实现。

##### 像素洗牌（Pixel Shuffle）支持
在《[开源的轻量化VLM-SmolVLM模型架构、数据策略及其衍生物PDF解析模型SmolDocling](https://mp.weixin.qq.com/s/2ZQKauOyMCDdXkzbFoDMhw)》提到，像素洗牌通过重新排列编码图像，以增加通道深度为代价换取空间分辨率。这减少了视觉标记数量，同时保持信息密度。

## 训练
训练数据得益于开源社区的快速发展，如FineVision、《[多模态视觉语言模型：Molmo2训练数据、训练配方](https://mp.weixin.qq.com/s/qquxlWRSRokP-SQrwyK_Dg)》提到的若干优质的数据集，结合一些筛选和净化手段。

训练整体分预训练和SFT两阶段：
- 预训练：训练模型的对齐能力，由VQA+OCR+caption数据构成。1024x1024低分辨率训练。
- SFT：训练模型的多模态理解能力，由纯文本+VQA的混合数据进行训练，2048x2048高分辨率训练。

## 推理代码

```
import torch
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor


model_dir = "模型权重"

model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)


tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
image_processor = CLIPImageProcessor.from_pretrained(model_dir, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "描述一下这张图片。"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "test.png"
                },
            }
        ],
    },
]
res = model.chat(messages, tokenizer, image_processor, max_new_tokens=1024, do_sample=True, temperature=0.6)
print(res)

```










# 2025.01
详细介绍：https://mp.weixin.qq.com/s/CH5FoRxoN6WHXPOMwG9gDA

- modelscrope：https://modelscope.cn/models/yujunhuinlp/Reyes-8B
- github：https://github.com/yujunhuics/Reyes



## 推理


使用方式：将本仓库中的`modeling_reyes.py`文件替换modelscrope下载的`modeling_reyes.py`运行即可。
batch推理详细见：`batch_inference.ipynb`




- 串行推理

```python
import torch
from modelscope import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def preprocess_image(file_path, dynamic=True, max_num=6, image_size=448):
    try:
        if dynamic:
            return load_image(file_path, max_num=max_num).to(torch.bfloat16).cuda()
        else:
            img = Image.open(file_path).convert('RGB')
            transform = build_transform(image_size)
            pixel_values = transform(img)
            return torch.stack([pixel_values]).to(torch.bfloat16).cuda()
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")


path = "Reyes-8B"

model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()

# print(model)

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=2048, do_sample=False)

# single-image single-round conversation
file_path = 'tmp.png'
pixel_values = preprocess_image(file_path, dynamic=True)
question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')

# pure-text conversation
question = 'Hello, who are you?'
response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

```


## 图片token化
```python
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=2, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=1):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def convert_image_token(image):
    if dynamic_image_size:
        image = Image.open(image).convert('RGB')
        num_tile = len(dynamic_preprocess(image))
        tile_pos_identifiers = [f"<tile_{i}>" for i in range(1, num_tile)] + ["<tile_global_thumbnail>"]
        image_tokens = ''
        for tile_pos_identifier in tile_pos_identifiers:
            image_tokens += tile_pos_identifier + IMG_CONTEXT_TOKEN * num_image_token
        image_tokens = IMG_START_TOKEN + image_tokens + IMG_END_TOKEN
    else:
        image_tokens = IMG_CONTEXT_TOKEN * num_image_token
        image_tokens = IMG_START_TOKEN + image_tokens + IMG_END_TOKEN
    return image_tokens


if __name__ == '__main__':
    IMG_START_TOKEN = '<|vision_start|>'
    IMG_CONTEXT_TOKEN = '<|vision_pad|>'
    IMG_END_TOKEN = '<|vision_end|>'

    force_image_size = 488
    down_sample_ratio = 0.5
    dynamic_image_size = True
    num_image_token = 256
    imagetokens = convert_image_token('test.png')
    print(imagetokens)
```
