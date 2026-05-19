from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer, get_model_config
import torch
import torch.utils.data as data
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


image_root = 'Datasets/Urban1k/image/'
#image_root = 'Datasets/Decay_Test/image/' # --> New Edits - 19-05-2026
caption_root = 'Datasets/Urban1k/caption/'
#caption_root = 'Datasets/Decay_Test/caption/' # --> New Edits - 19-05-2026

class local_dataset(data.Dataset):
    def __init__(self, data_path):
        self.image_root = f"{data_path}{image_root}"
        self.caption_root = f"{data_path}{caption_root}"
        self.total_image = os.listdir(self.image_root)
        self.total_caption = os.listdir(self.caption_root)

    def __len__(self):
        return len(self.total_caption)

    def __getitem__(self, index):
        caption_name = self.total_caption[index]
        image_name = self.total_caption[index][:-4] + '.jpg'
        image = Image.open(self.image_root + image_name)
        f=open(self.caption_root + caption_name)
        caption = f.readlines()[0]
        
        return image, caption

class OptimizedLocalDataset(data.Dataset):
    def __init__(self, data_path, processor):
        self.image_root = f"{data_path}{image_root}"
        self.caption_root = f"{data_path}{caption_root}"
        self.total_image = os.listdir(self.image_root)
        self.total_caption = os.listdir(self.caption_root)
        self.processor = processor    
    def __len__(self):
        return len(self.total_caption)
    
    def __getitem__(self, index):
        caption_name = self.total_caption[index]
        image_name = self.total_caption[index][:-4] + '.jpg'

        with Image.open(self.image_root + image_name) as img:
            img_tensor = self.processor(img)

        with open(self.caption_root + caption_name) as f:
            caption = f.readlines()[0]

        return img_tensor, caption
    

def run_urban1k_openclip(model, distilled_model, processor, data_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    distilled_model.eval()

    print("\n=== Distilled Model Info ===") # --> New Edits - 15-05-2026
    print(distilled_model) # --> New Edits - 15-05-2026

    # Create DataLoader
    batch_size = 16  # Adjust based on your GPU memory
    dataset = OptimizedLocalDataset(data_path, processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    img_feature_list = []
    text_feature_list = []

    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Processing batches"):
            # Process batch of images
            images = images.to(device)
            image_features = model.encode_image(images)
            img_feature_list.append(image_features)

            # Process batch of captions
            text_encoded = processor.tokenizer(captions).to(device)
            text_features = distilled_model.encode_text(text_encoded)
            text_feature_list.append(text_features)

        # Concatenate all features
        image_embeds = torch.cat(img_feature_list, dim=0)
        text_feature = torch.cat(text_feature_list, dim=0)

        # Normalize features
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        #Getting saved Q and K
        print("Getting saved Q and K") # --> New Edits - 18-05-2026
        attn_layer = distilled_model.transformer.resblocks[0].attn # --> New Edits - 18-05-2026

        if not hasattr(attn_layer, 'saved_q'): # --> New Edits - 18-05-2026 
            print("Error: Could not find 'saved_q'. Make sure you added the two lines to AttentionRoPE!") # --> New Edits - 18-05-2026
            return # --> New Edits - 18-05-2026

        q_rot = attn_layer.saved_q  # Shape: (N, L, num_heads, head_dim) # --> New Edits - 18-05-2026
        k_rot = attn_layer.saved_k  # Shape: (N, L, num_heads, head_dim) # --> New Edits - 18-05-2026

        print(f"Shape of saved_q: {q_rot.shape}") # --> New Edits - 18-05-2026
        print(f"Shape of saved_k: {k_rot.shape}") # --> New Edits - 18-05-2026

        # Isolate the first batch (0) and first attention head (0)
        q_head = q_rot[0, :, 0, :] # Shape: [L, head_dim] # --> New Edits - 18-05-2026
        k_head = k_rot[0, :, 0, :] # Shape: [L, head_dim] # --> New Edits - 18-05-2026

        print(f"Shape of first attention head q: {q_head.shape}") # --> New Edits - 18-05-2026
        print(f"Shape of first attention head k: {k_head.shape}") # --> New Edits - 18-05-2026

        # Calculate Attention Scores
        # We want to see how token 0 (Query 0) attends to all other tokens (Keys 0 to L) # --> New Edits - 18-05-2026
        #q0 = q_head[0, :] # The query for the very first token # --> New Edits - 18-05-2026
        q0 = q_head[5, :] # The query for the very first token # --> New Edits - 19-05-2026 (for plotting try)

        # Dot product of q0 with all keys: [head_dim] @ [L, head_dim].T -> [L] # --> New Edits - 18-05-2026
        print("Getting attention scores") # --> New Edits - 18-05-2026
        #attention_scores = torch.matmul(q0, k_head.transpose(-1, -2)).cpu().numpy() # --> New Edits - 18-05-2026
        attention_scores = torch.matmul(q0, k_head[5:240, :].transpose(-1, -2)).cpu().numpy() # --> New Edits - 19-05-2026 (for plotting try)
        print(type(attention_scores), attention_scores.shape, attention_scores.size) # --> New Edits - 18-05-2026

        # Plot results
        plt.figure(figsize=(10, 5), dpi=120) # --> New Edits - 18-05-2026
        plt.plot(range(len(attention_scores)), attention_scores, label='Attention Score (RoPE Decay)', color='blue') # --> New Edits - 18-05-2026
    
        plt.title('RoPE Positional Decay (Extracted from Trained Model)') # --> New Edits - 18-05-2026
        plt.xlabel('Token Distance', fontweight='bold') # --> New Edits - 18-05-2026
        plt.ylabel('Pre-Softmax Attention Score', fontweight='bold') # --> New Edits - 18-05-2026
        plt.grid(True, linestyle='--', alpha=0.6) # --> New Edits - 18-05-2026
        plt.legend() # --> New Edits - 18-05-2026
        plt.tight_layout() # --> New Edits - 18-05-2026
        #plt.show() # --> New Edits - 18-05-2026
        plt.savefig('rope_decay_plot.png', bbox_inches='tight') # --> New Edits - 18-05-2026
        print("Plot successfully saved as 'rope_decay_plot.png'!") # --> New Edits - 18-05-2026
        plt.close() # --> New Edits - 18-05-2026

        logit_scale = 100
        
        # Calculate metrics
        metrics = get_clip_metrics(image_embeds, text_feature, logit_scale)
        
        # Print metrics
        for k in [1, 5, 10]:
            print(f"Text to Image - R@{k}: {metrics[f'text_to_image_R@{k}']}")
            print(f"Image to Text - R@{k}: {metrics[f'image_to_text_R@{k}']}")

        return metrics

def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
