# 라이브러리 불러오기
import os
import csv
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


# 파일 경로
data_dir = './flickr30k'
captions_file = os.path.join(data_dir, 'captions.txt')

# 데이터 초기화
captions_dict = {}
image_list = set()

# 데이터 구조화
with open(captions_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row['image_name']
        caption = row['comment']

        if filename not in captions_dict:
            captions_dict[filename] = []
        captions_dict[filename].append(caption)
        image_list.add(filename)

# 이미지 리스트 생성
image_list = list(image_list)

# train/test 데이터셋 나누기
train_images, test_images = train_test_split(image_list, test_size=0.02, random_state=42)

# 이미지-캡션 매칭
train_captions = {img: captions_dict[img] for img in train_images}
test_captions = {img: captions_dict[img] for img in test_images}

# test용 이미지 리스트 저장
test_images = list(test_captions.keys())

def calculate_similarity(image_path, captions):
    # 전처리
    image = Image.open(image_path)
    inputs = processor(
        text=captions,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # 모델 예측 (유사도 점수를 확률로 계산)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # 엔트로피 계산
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).item()

    return probs, entropy




class Flickr30kDataset(Dataset):
    def __init__(self, image_caption_pairs, processor, images_dir):
        self.image_caption_pairs = image_caption_pairs
        self.processor = processor
        self.images_dir = images_dir

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        img_filename, caption = self.image_caption_pairs[idx]
        img_path = os.path.join(self.images_dir, img_filename)
    
        # 이미지 로드 및 전처리
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True  # 긴 입력 시퀀스를 자르도록 설정
        )
    
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }


class PreferenceDataset(Dataset):
    def __init__(self, captions_dict, processor, images_dir, mode="DPO", max_text_length=77):
        self.processor = processor
        self.images_dir = images_dir
        self.mode = mode
        self.max_text_length = max_text_length

        self.image_caption_pairs = []
        for img, captions in captions_dict.items():
            captions = [cap for cap in captions if len(cap.split()) <= self.max_text_length]

            if len(captions) < 2:
                continue
            img_path = os.path.join(images_dir, img)

            if mode == "DPO":
                pairs = [(img_path, captions[i], captions[j]) for i in range(len(captions)) for j in range(len(captions)) if i != j]
            elif mode == "TPO":
                pairs = [(img_path, captions[i], captions[j], captions[k]) for i in range(len(captions)) for j in range(len(captions)) for k in range(len(captions)) if i != j and j != k and i != k]
            self.image_caption_pairs.extend(pairs)

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        pair = self.image_caption_pairs[idx]
        image = Image.open(pair[0]).convert("RGB")
        if self.mode == "DPO":
            caption1, caption2 = pair[1], pair[2]
            inputs = self.processor(
                text=[caption1, caption2],
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length
            )
            return inputs
        elif self.mode == "TPO":
            caption1, caption2, caption3 = pair[1], pair[2], pair[3]
            inputs = self.processor(
                text=[caption1, caption2, caption3],
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length
            )
            return inputs


# collate 함수
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"].squeeze(0) for item in batch])
    input_ids = torch.cat([item["input_ids"].squeeze(0) for item in batch], dim=0)
    attention_mask = torch.cat([item["attention_mask"].squeeze(0) for item in batch], dim=0)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


# 손실 함수
def tpo_loss(logits, temperature=1.0):
    pos_score, neg_score, alt_score = logits[:, 0], logits[:, 1], logits[:, 2]
    numerator = torch.exp(pos_score / temperature)
    denominator = torch.exp(pos_score / temperature) + torch.exp(neg_score / temperature) + torch.exp(alt_score / temperature)
    return -torch.log(numerator / denominator).mean()

# 데이터로더 생성
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tpo_dataset = PreferenceDataset(train_captions, processor, os.path.join(data_dir, "flickr30k_images"), mode="TPO")
tpo_dataloader = DataLoader(
    tpo_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn
)



# 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 5

# 학습
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress = tqdm(tpo_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (TPO)")
    for batch in progress:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits_per_image
        loss = tpo_loss(logits)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress.set_postfix({"Loss": loss.item()})

    # 모델 저장
    output_dir = f"./clip_tpo_finetune/clip_finetuning_tpo_{epoch}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"TPO Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(tpo_dataloader):.4f}")
