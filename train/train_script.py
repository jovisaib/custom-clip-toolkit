import torch
from PIL import Image


from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import clip


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
list_image_paths = []
list_labels = []

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = clip.tokenize(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = preprocess(Image.open(image_path).convert("RGB"))
        return image, label

dataset = CustomDataset(list_image_paths, list_labels)
dataloader = DataLoader(dataset, batch_size=120, shuffle=True)

if device == "cpu":
    model.float()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9,0.98), eps=1e-6, weight_decay=1e-5)
loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()

num_epochs = 8


def train():
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        for batch in pbar:
            optimizer.zero_grad()
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            logits_per_image, logits_per_text = model(images, labels)

            # Compute loss
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

            # Backward pass
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else :
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")