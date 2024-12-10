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

# CLIP, Processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


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


# train/test 데이터셋 나누기
train_pairs = [(img, cap) for img, captions in train_captions.items() for cap in captions]
train_dataset = Flickr30kDataset(train_pairs, processor, os.path.join(data_dir, "flickr30k_images"))

test_pairs = [(img, cap) for img, captions in test_captions.items() for cap in captions]
test_dataset = Flickr30kDataset(test_pairs, processor, os.path.join(data_dir, "flickr30k_images"))


# collate 함수
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

# train/test 데이터로더 생성
train_dataloader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
)

test_dataloader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 옵티마이저, 손실함수
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()


# 학습
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_progress = tqdm(train_dataloader, desc="Training")

    # 데이터 전처리
    for batch in train_progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        # 모델 출력
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_loss=True
        )

        # 손실 계산
        loss = outputs.loss
        total_loss += loss.item()

        # 역전파 & 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_progress.set_postfix({"Loss": loss.item()})


    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_dataloader):.4f}")

    # 결과 출력 & 모델 저장
    output_dir = f"./clip_finetune2/clip_finetuning_{epoch}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)



