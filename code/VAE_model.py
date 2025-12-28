import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualConvBlock(nn.Module):
    """Conv block with residual connection using Instance Normalization"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(out_ch, affine=True)  # affine=True to learn scale/shift
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_ch, affine=True)

        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = x if self.res_conv is None else self.res_conv(x)
        x = self.relu(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        x += residual
        return self.relu(x)

class ResidualConvBlock2(nn.Module):
    """Conv block with residual connection using Instance Normalization"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in1 = nn.InstanceNorm2d(in_ch, affine=True)  # affine=True to learn scale/shift
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.in2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = x if self.res_conv is None else self.res_conv(x)
        x = self.conv1(self.relu(self.in1(x)))
        x = self.conv2(self.relu2(self.in2(x)))
        x += residual
        return x

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)  # pt = probability of true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def cyclical_kl_weight(epoch, cycle_length=30, max_weight=0.01):
        # epoch % cycle_length goes from 0 → cycle_length-1
        cycle_pos = epoch % cycle_length
        return max_weight * (cycle_pos / (cycle_length // 2)) if cycle_pos < (cycle_length // 2) \
            else max_weight * (1 - (cycle_pos - cycle_length // 2) / (cycle_length // 2))

def linear_kl_weight(step, total_steps, warmup_fraction=0.2, beta_max=0.03, beta_min=0.0):
    """
    Linear warmup of KL weight β for VAEs.
    - warmup_fraction: fraction of training steps used to ramp up β
    - beta_max: final β value
    - beta_min: starting β (usually 0)
    """
    warmup_steps = int(total_steps * warmup_fraction)
    if warmup_steps <= 0:
        return beta_max
    if step < warmup_steps:
        return beta_min + (beta_max - beta_min) * (step / warmup_steps)
    return beta_max

def generalized_dice_loss(pred, target, class_volumes=None, eps=1e-6):
    """
    pred: (B,C,H,W) - predicted logits
    target: (B,H,W) - target labels
    class_volumes: (C,) - pre-computed class volumes over entire dataset (optional)
    """
    pred_soft = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0,3,1,2).float()
    
    if class_volumes is not None:
        w = 1.0 / (class_volumes + eps)**2
        w = w / w.sum() * pred.shape[1]  # normalize weights
        w = w.to(pred.device)
    else:
        w = 1.0 / (torch.sum(target_onehot, dim=(0,2,3))**2 + eps)

    numerator = 2 * torch.sum(w * torch.sum(pred_soft * target_onehot, dim=(0,2,3)))
    denominator = torch.sum(w * torch.sum(pred_soft + target_onehot, dim=(0,2,3)))
    loss = 1.0 - numerator / (denominator + eps)
    return loss


def soft_dice_loss(pred, target, eps=1e-6):
    """
    pred: logits (B, C, H, W)
    target: (B, H, W) with integer class labels
    computes mean(1 - dice) where dice is computed per-sample and averaged
    """
    pred_soft = torch.softmax(pred, dim=1)  # (B, C, H, W)
    num_classes = pred_soft.shape[1]
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0,3,1,2).float()  # (B, C, H, W)

    # compute per-sample per-class intersection and denominator
    intersection = torch.sum(pred_soft * target_onehot, dim=(2,3))  # (B, C)
    denominator = torch.sum(pred_soft + target_onehot, dim=(2,3))  # (B, C)

    dice_per_class = (2 * intersection + eps) / (denominator + eps)  # (B, C)
    # mean over classes, then mean over batch
    dice_per_sample = dice_per_class.mean(dim=1)  # (B,)
    dice = dice_per_sample.mean()  # scalar
    return 1.0 - dice  # dice loss

class ResidualVAE_Segmenter(nn.Module):
    def __init__(self, latent_dim=128, n_classes=4, dropout=0.1, v_ce=0.25, v_dice=0.25, v_recon=0.5):
        super().__init__()
        self.v_ce = v_ce
        self.v_dice = v_dice
        self.v_recon = v_recon
        self.n_classes = n_classes
        self.class_volumes = None  # Will be computed from dataset
        print("Initializing model... VAE_model_up")
        # ---------------- Encoder ----------------
        self.enc1 = nn.Sequential(
            ResidualConvBlock2(4, 32),
            ResidualConvBlock2(32, 32)
        )
        self.enc2 = nn.Sequential(
            ResidualConvBlock2(32, 64),
            ResidualConvBlock2(64, 64)
        )
        self.enc3 = nn.Sequential(
            ResidualConvBlock2(64, 128),
            ResidualConvBlock2(128, 128)
        )
        self.enc4 = nn.Sequential(
            ResidualConvBlock2(128, 128),
            ResidualConvBlock2(128, 128)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((15,15))
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128*15*15, latent_dim)
        self.fc_logvar = nn.Linear(128*15*15, latent_dim)
        
        # Separate latent space projections for each decoder (expand back to 128*15*15)
        self.fc_decode_recon = nn.Linear(latent_dim, 128*15*15)
        self.fc_decode_seg = nn.Linear(latent_dim, 128*15*15)

        # ---------------- Reconstruction Decoder ----------------
        self.recon_up1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.recon_dec1 = ResidualConvBlock2(128, 128)
        self.recon_up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.recon_dec2 = ResidualConvBlock2(64, 64)
        self.recon_up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.recon_dec3 = ResidualConvBlock2(32, 32)
        # additional up block to exactly recover 240x240 from 15x15 bottleneck
        self.recon_up4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.recon_dec4 = ResidualConvBlock2(16, 16)
        self.recon_drop = nn.Dropout2d(dropout)
        self.recon_head = nn.Conv2d(16, 4, kernel_size=3, padding=1)

        # ---------------- Segmentation Decoder ----------------
        self.seg_up1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.seg_dec1 = ResidualConvBlock2(128, 128)
        self.seg_up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.seg_dec2 = ResidualConvBlock2(64, 64)
        self.seg_up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.seg_dec3 = ResidualConvBlock2(32, 32)
        self.seg_up4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.seg_dec4 = ResidualConvBlock2(16, 16)
        self.seg_drop = nn.Dropout2d(dropout)
        self.seg_head = nn.Conv2d(16, n_classes, kernel_size=3, padding=1)

        print(f"Model initialized with {self.calculate_model_parameters()/1e6:.2f}M parameters.")
    # ---------------- Forward ----------------
    def encode(self, x):
        x = F.max_pool2d(self.enc1(x), 2) # pool is a dimension reducer
        x = F.max_pool2d(self.enc2(x), 2)
        x = F.max_pool2d(self.enc3(x), 2)
        x = F.max_pool2d(self.enc4(x), 2)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode_reconstruction(self, z, output_size):
        """Separate decoder for reconstruction task"""
        x = self.fc_decode_recon(z)
        x = x.view(-1, 128, 15, 15)

        x = self.recon_up1(x)
        x = self.recon_dec1(x)
        x = self.recon_drop(x)

        x = self.recon_up2(x)
        x = self.recon_dec2(x)
        x = self.recon_drop(x)

        x = self.recon_up3(x)
        x = self.recon_dec3(x)
        x = self.recon_drop(x)

        x = self.recon_up4(x)
        x = self.recon_dec4(x)
        x = self.recon_drop(x)

        recon_output = self.recon_head(x)
        
        # Upsample output to match original input size
        #recon_output = F.interpolate(recon_output, size=output_size, mode="bilinear", align_corners=False)
        return recon_output

    def decode_segmentation(self, z, output_size):
        """Separate decoder for segmentation task"""
        x = self.fc_decode_seg(z)
        x = x.view(-1, 128, 15, 15)

        x = self.seg_up1(x)
        x = self.seg_dec1(x)
        x = self.seg_drop(x)

        x = self.seg_up2(x)
        x = self.seg_dec2(x)
        x = self.seg_drop(x)

        x = self.seg_up3(x)
        x = self.seg_dec3(x)
        x = self.seg_drop(x)

        x = self.seg_up4(x)
        x = self.seg_dec4(x)
        x = self.seg_drop(x)


        seg_output = self.seg_head(x)
        
        # Upsample output to match original input size
        #seg_output = F.interpolate(seg_output, size=output_size, mode="bilinear", align_corners=False)
        return seg_output

    def decode(self, z, output_size):
        """Combined decode method for backward compatibility"""
        seg_output = self.decode_segmentation(z, output_size)
        recon_output = self.decode_reconstruction(z, output_size)
        return seg_output, recon_output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output_size = x.shape[2:]  # H, W
        seg_output, recon_output = self.decode(z, output_size)
        return seg_output, recon_output, mu, logvar

    def calculate_model_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params

    def compute_class_volumes(self, dataloader):
        """
        Compute class volumes over entire dataset for generalized dice loss
        """
        print("Computing class volumes from dataset...")
        self.eval()
        class_counts = torch.zeros(self.n_classes, dtype=torch.float32)
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                _, y, _ = batch
                y = y.long()
                
                # Convert to one-hot and count pixels per class
                target_onehot = F.one_hot(y, num_classes=self.n_classes).float()  # (B, H, W, C)
                batch_counts = torch.sum(target_onehot, dim=(0, 1, 2))  # (C,)
                class_counts += batch_counts.cpu()
                total_samples += y.numel()
        
        # Store class volumes (total pixels per class across dataset)
        self.class_volumes = class_counts
        
        print(f"Class volumes computed from {total_samples:,} pixels:")
        for i, volume in enumerate(class_counts):
            percentage = (volume / total_samples) * 100
            print(f"  Class {i}: {volume:,.0f} pixels ({percentage:.1f}%)")
        
        return self.class_volumes


    # ---------------- Loss ----------------
    def loss_fn(self, pred_seg_logits, target_seg, pred_recon, input_img, mu, logvar,
                epoch=None, kl_max_weight=0.03, kl_anneal_epochs=50):
        # Ensure target_seg is the correct type (Long tensor)
        target_seg = target_seg.long()
        # Ensure spatial dimensions match between predictions and targets
        try:
            target_hw = target_seg.shape[1:]
            pred_hw = pred_seg_logits.shape[2:]
            if pred_hw != target_hw:
                # Resize logits and reconstruction to match target size
                # Interpolating logits is acceptable for segmentation loss computation
                print(f"[LOSS] Resizing model outputs from {pred_hw} to target {target_hw} to match target for loss computation")
                pred_seg_logits = F.interpolate(pred_seg_logits, size=target_hw, mode='bilinear', align_corners=False)
                # Also ensure reconstruction matches input image size
                input_hw = input_img.shape[2:]
                if pred_recon.shape[2:] != input_hw:
                    pred_recon = F.interpolate(pred_recon, size=input_hw, mode='bilinear', align_corners=False)
        except Exception:
            # If something unexpected happens, continue and let downstream errors surface
            pass
        
        #ce_loss = F.cross_entropy(pred_seg_logits, target_seg)
        focal = FocalLoss(gamma=3.0)  # you can tune gamma (1–3 typical)
        ce_loss = focal(pred_seg_logits, target_seg)

        pred_seg_probs = torch.softmax(pred_seg_logits, dim=1)
        target_seg_1hot = F.one_hot(target_seg, num_classes=pred_seg_probs.shape[1]).permute(0,3,1,2).float()
        intersection = 2.0 * torch.sum(pred_seg_probs * target_seg_1hot, dim=(0,2,3))
        denominator = torch.sum(pred_seg_probs + target_seg_1hot, dim=(0,2,3)) + 1e-6
        #dice_loss = 1.0 - torch.mean(intersection / denominator)
        dice_loss = generalized_dice_loss(pred_seg_logits, target_seg, self.class_volumes)
        #dice_loss = soft_dice_loss(pred_seg_logits, target_seg)

        l2_loss = F.mse_loss(pred_recon, input_img, reduction='mean')

        logvar = torch.clamp(logvar, -10, 10)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        if epoch is not None:
            kl_weight = linear_kl_weight(epoch, 100) #cyclical_kl_weight(epoch, cycle_length=kl_anneal_epochs, max_weight=kl_max_weight)
        else:
            kl_weight = kl_max_weight


        total_loss = self.v_ce*ce_loss + self.v_dice*dice_loss + self.v_recon*l2_loss + kl_weight*kl_loss
        #total_loss = 0.2*ce_loss + 0.25*dice_loss + 0.5*l2_loss + kl_weight*kl_loss

        return total_loss, {"dice": dice_loss.item(), "ce": ce_loss.item(),
                            "recon": l2_loss.item(), "kl": kl_loss.item(), "kl_weight": kl_weight}

    # ---------------- Training Step ----------------
    def training_step(self, batch, optimizer, epoch=None):
        self.train()
        optimizer.zero_grad()
        X, y, _ = batch
        device = next(self.parameters()).device
        X, y = X.to(device), y.to(device)
        pred_seg, pred_recon, mu, logvar = self(X)
        loss, logs = self.loss_fn(pred_seg, y, pred_recon, X, mu, logvar, epoch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

        optimizer.step()
        print(logs)
        return loss.item(), logs
    @staticmethod
    def compute_hard_dice(pred_logits, target, num_classes, eps=1e-6):
        """
        Compute per-class hard Dice for a batch
        pred_logits: (B, C, H, W)
        target: (B, H, W)
        Returns:
            dice_per_class: tensor of shape (C,)
        """
        preds = pred_logits.argmax(dim=1)  # (B, H, W)
        dice_per_class = torch.zeros(num_classes, device=pred_logits.device)

        for c in range(num_classes):
            pred_c = (preds == c).float()
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            denominator = pred_c.sum() + target_c.sum() + eps
            dice_per_class[c] = (2.0 * intersection + eps) / denominator

        return dice_per_class

    # ---------------- Training Loop ----------------
    def validate(self, val_dataloader, epoch=None, class_groups = {"NT": (1), "ED": (2), "ET": (3) , "B": (0)}):
        from collections import defaultdict
        """Validation step - no gradient updates"""
        self.eval()
        total_loss, total_logs = 0, {"dice":0,"ce":0,"recon":0,"kl":0,"kl_weight":0}
        total_dice_per_class = None
        total_samples = 0
        num_batches = 0
        num_classes = self.seg_head.out_channels  # your n_classes

        case_dice_acc = defaultdict(lambda: {"intersection": defaultdict(float),
                                         "union": defaultdict(float),
                                         "slice_count": 0})

        with torch.no_grad():
            for batch in val_dataloader:
                X, y, case_ids = batch
                batch_size = X.size(0)
                device = next(self.parameters()).device
                X, y = X.to(device), y.to(device)

                mu, logvar = self.encode(X)
                pred_seg, pred_recon = self.decode(mu, None)
                pred_label = torch.argmax(pred_seg, dim=1)
                loss, logs = self.loss_fn(pred_seg, y, pred_recon, X, mu, logvar, epoch)

                total_loss += loss.item() * batch_size
                for k in total_logs:
                    total_logs[k] += logs.get(k, 0) * batch_size

                # Compute hard Dice per batch
                batch_dice_per_class = self.compute_hard_dice(pred_seg, y, num_classes)
                if total_dice_per_class is None:
                    total_dice_per_class = batch_dice_per_class * batch_size
                else:
                    total_dice_per_class += batch_dice_per_class * batch_size

                total_samples += batch_size
                num_batches += 1

                # accumulate intersection & union per class per case
                for i, cid in enumerate(case_ids):
                    pred_slice = pred_label[i].cpu()
                    gt_slice = y[i].cpu()

                    for group_name, class_idx in class_groups.items():
                        pred_mask = torch.isin(pred_slice, torch.tensor(class_idx))
                        gt_mask = torch.isin(gt_slice, torch.tensor(class_idx))

                        case_dice_acc[cid]["intersection"][group_name] += (pred_mask & gt_mask).sum().item()
                        case_dice_acc[cid]["union"][group_name] += pred_mask.sum().item() + gt_mask.sum().item()

                    case_dice_acc[cid]["slice_count"] += 1
        
        # ---------------- Compute 3D Dice per case ----------------
        final_case_dice = {}
        for cid, stats in case_dice_acc.items():
            dice_case = {}
            for group_name in class_groups.keys():
                inter = stats["intersection"][group_name]
                union = stats["union"][group_name]
                dice_case[group_name] = 1.0 if union == 0 else 2 * inter / union
            dice_case["mean_dice"] = sum(dice_case.values()) / len(class_groups)
            final_case_dice[cid] = dice_case

        # ---------------- Average over all cases ----------------
        mean_scores = {k: 0.0 for k in list(class_groups.keys()) + ["mean_dice"]}
        for dice in final_case_dice.values():
            for k, v in dice.items():
                mean_scores[k] += v
        num_cases = len(final_case_dice)
        for k in mean_scores:
            mean_scores[k] /= num_cases

        avg_loss = total_loss / total_samples
        avg_logs = {k:v/total_samples for k,v in total_logs.items()}
        avg_dice_per_class = total_dice_per_class / total_samples
        avg_logs['hard_dice'] = avg_dice_per_class.mean().item()  # optional: mean across classes

        print(f"    [VALIDATION DEBUG] Processed {total_samples} slices in {num_batches} batches")
        print(f"    [VALIDATION HARD DICE] {avg_dice_per_class.tolist()}")
        print(f"    [VALIDATION 3D HARD DICE] {mean_scores}")

        return avg_loss, avg_logs, mean_scores


    def training_loop(self, dataloader, optimizer, scheduler, epochs=100, save_best=True, val_dataloader=None):
        import time
        best_dice = float('inf')  # Lower dice loss is better
        best_epoch = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            total_loss, total_logs = 0, {"dice":0,"ce":0,"recon":0,"kl":0,"kl_weight":0}
            for batch in dataloader:
                loss, logs = self.training_step(batch, optimizer, epoch)
                total_loss += loss
                for k in total_logs:
                    total_logs[k] += logs.get(k, 0)
            train_avg_loss = total_loss / len(dataloader)
            train_avg_logs = {k:v/len(dataloader) for k,v in total_logs.items()}
            
            # Validation phase
            val_info = ""
            if val_dataloader is not None:
                val_avg_loss, val_avg_logs, dice_3d = self.validate(val_dataloader, epoch)
                val_info = f" | Val Loss: {val_avg_loss:.4f}, Val Dice: {val_avg_logs['dice']:.4f}"
                
                # Use validation dice for best model selection if available
                current_dice = 1 - dice_3d["mean_dice"] #val_avg_logs['dice']
            else:
                # Fall back to training dice if no validation set
                current_dice = train_avg_logs['dice']
            
            # Check if this is the best model so far (keep in memory only)
            if save_best and current_dice < best_dice:
                best_dice = current_dice
                best_epoch = epoch + 1
                best_model_state = self.state_dict().copy()  # Store best weights in memory
                print(f"🏆 NEW BEST! Epoch {epoch+1}, Dice: {current_dice:.4f}")
            
            print(f"Epoch {epoch+1}, Train Loss: {train_avg_loss:.4f}, Train Logs: {train_avg_logs}{val_info}")
            print(f"    Best so far: Epoch {best_epoch}, Dice: {best_dice:.4f}")
            
            # Handle different scheduler types
            if hasattr(scheduler, 'step') and 'ReduceLROnPlateau' in str(type(scheduler)):
                scheduler.step(train_avg_loss)  # ReduceLROnPlateau needs the loss value
            else:
                scheduler.step()  # Other schedulers don't need loss value
        
        # Save models only at the end of training
        if save_best and best_model_state is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            best_path = f"vae_segmenter_best_{timestamp}_epoch{best_epoch}_dice{best_dice:.4f}.pth"
            torch.save(best_model_state, best_path)
            print(f"💾 Saved best model: {best_path}")
            
            # Also save current (final) model for comparison
            final_path = f"vae_segmenter_final_{timestamp}.pth"
            torch.save(self.state_dict(), final_path)
            print(f"💾 Saved final model: {final_path}")
            
            return best_path, final_path
        
        return None, None

    def save_model(self, path):
        torch.save(self.state_dict(), path)