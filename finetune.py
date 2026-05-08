import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import time
import skimage.color

# -----------------------------------------------------------------------------
#  Dataset
# -----------------------------------------------------------------------------
class ColorDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files
        from colorizers import preprocess_img
        self.preprocess = preprocess_img

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        t_orig, t_rs = self.preprocess(img_rgb, HW=(256, 256))
        # We now use skimage for correct LAB conversion.
        # Ensure img_rgb is scaled to 0-1 for skimage.color.rgb2lab or just pass it if it handles 0-255.
        # skimage.color.rgb2lab expects 0-255 if it's uint8, or 0-1 if float.
        img_lab_np = skimage.color.rgb2lab(img_rgb)
        img_lab = torch.from_numpy(img_lab_np).permute(2, 0, 1).float()
        return t_rs.squeeze(0), t_orig.squeeze(0), img_lab[1:, :, :], torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.0 # (L_rs, L_orig, ab_target, rgb_target)

# -----------------------------------------------------------------------------
#  UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Model Fine-Tuner", page_icon="🧪", layout="wide")

st.markdown("# 🧪 Model Fine-Tuning Suite")
st.markdown("Re-train the decoder weights on your custom images for better domain-specific results.")

with st.sidebar:
    st.markdown("## ⚙️ Hyperparameters")
    model_type = st.selectbox("Model to fine-tune", ["ECCV-16", "SIGGRAPH-17", "ViT-B/16"])
    lr = st.number_input("Learning Rate", value=1e-5, format="%.1e")
    epochs = st.number_input("Epochs", value=5, min_value=1)
    batch_size = st.number_input("Batch Size", value=1, min_value=1)
    
st.info("💡 To start, upload 1 or more colour reference images (e.g. historical portraits). The model will learn to colorize their B&W versions correctly.")

uploaded_files = st.file_uploader("Upload reference images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 Start Fine-Tuning", type="primary"):
        # Save temp files
        os.makedirs("tmp_train", exist_ok=True)
        img_paths = []
        for f in uploaded_files:
            p = os.path.join("tmp_train", f.name)
            with open(p, "wb") as wb:
                wb.write(f.getbuffer())
            img_paths.append(p)
            
        st.write(f"Preparing dataset with {len(img_paths)} images...")
        dataset = ColorDataset(img_paths)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Load model
        from colorizers import eccv16, siggraph17, eccv16_upgraded
        if model_type == "ECCV-16":
            model = eccv16(pretrained=True).train()
        elif model_type == "SIGGRAPH-17":
            model = siggraph17(pretrained=True).train()
        else:
            model = eccv16_upgraded(backbone="vit").train()
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Load our new custom composite loss
        from colorizers.custom_losses import ColorizationLoss
        criterion = ColorizationLoss(lambda_smooth_l1=1.0, lambda_vgg=0.1, lambda_ciede=0.5).to(device)
        
        # Differentiable LAB to RGB
        def lab_to_rgb_tensor(l, ab):
            """Simplified differentiable LAB to RGB for perceptual loss"""
            L = (l + 50.0) # assuming l is -50 to 50
            # Normalize to 0-1 for a rough approximation to pass into VGG
            # This avoids complex color space math but gives the VGG a reasonable 3-channel input
            # VGG is robust enough to learn perceptual features from this pseudo-RGB
            r = L / 100.0 + ab[:, 0:1, :, :] / 110.0
            g = L / 100.0 - ab[:, 0:1, :, :] / 110.0 - ab[:, 1:2, :, :] / 110.0
            b = L / 100.0 + ab[:, 1:2, :, :] / 110.0
            rgb = torch.cat([r, g, b], dim=1)
            return torch.clamp(rgb, 0, 1)

        progress = st.progress(0)
        status = st.empty()
        
        for epoch in range(epochs):
            total_loss = 0
            for i, (L_rs, L_orig, ab_target, rgb_target) in enumerate(loader):
                optimizer.zero_grad()
                
                L_rs = L_rs.to(device)
                ab_target = ab_target.to(device)
                rgb_target = rgb_target.to(device)
                
                # Mock forward for simplicity
                if model_type == "ViT-B/16":
                    out = model(L_rs)
                    pred_ab = out["ab"]
                else:
                    pred_ab = model(L_rs)
                
                # Rescale target to match output if needed
                target_ab = torch.nn.functional.interpolate(ab_target, size=pred_ab.shape[2:])
                target_rgb = torch.nn.functional.interpolate(rgb_target, size=pred_ab.shape[2:])
                
                # We need L channel at the prediction resolution
                L_rs_resized = torch.nn.functional.interpolate(L_rs, size=pred_ab.shape[2:])
                
                pred_rgb = lab_to_rgb_tensor(L_rs_resized, pred_ab)
                target_rgb_approx = lab_to_rgb_tensor(L_rs_resized, target_ab) # Match domain
                
                loss, l_sl1, l_vgg, l_ciede = criterion(pred_ab, target_ab, pred_rgb, target_rgb_approx)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(loader)
            status.markdown(f"**Epoch {epoch+1}/{epochs}** - Loss: `{avg_loss:.4f}` (SL1: `{l_sl1.item():.4f}`, VGG: `{l_vgg.item():.4f}`, CIEDE: `{l_ciede.item():.4f}`)")
            progress.progress((epoch + 1) / epochs)
            
        st.success("✅ Fine-tuning complete! Weights saved to `finetuned_weights/` (simulated)")
        os.makedirs("finetuned_weights", exist_ok=True)
        torch.save(model.state_dict(), f"finetuned_weights/{model_type.lower().replace('-','')}_custom.pth")
