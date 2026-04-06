import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import cv2
from scipy.ndimage import label as scipy_label

DATA_PATH = "../Use-Case-datasets/CIFAR10/"
OUTPUT_PATH = "../XAI-Methods-outputs/CIFAR10/"

# Set publication-quality parameters for plots
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 24
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.width'] = 2

# CIFAR-10 normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])       # standard CIFAR-10 stats
])

# download CIFAR-10 dataset (if not already present)
trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

classes = trainset.classes  # ["airplane", "automobile", "bird", ...]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading pretrained CIFAR-10 model from PyTorch Hub...")
try:
    # This model is already trained on CIFAR-10 with ~92% accuracy
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    print("Successfully loaded pretrained CIFAR-10 ResNet20 model")
except Exception as e:
    print(f"Error loading from torch.hub: {e}")

model = model.to(device)
model.eval()

# 1. Forward pass: Feed the image through the network and save the activations of a target convolutional layer.
# 2. Backward pass: Compute the gradient of the class score w.r.t. these activations
# 3. Compute weights (α): Grad-CAM++ computes a weighted sum of the gradients using the formula in the code. This emphasizes important pixels while reducing noise.
# 4. Generate CAM: Multiply the weights by the activations and sum across channels to get a heatmap. This is the Class Activation Map (CAM)
# 5. Normalize: Scale the heatmap between 0 and 1 so it’s easier to visualize.
class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, class_idx):
        grads = self.gradients
        acts = self.activations
        
        # Grad-CAM++ weighting
        alpha_num = grads.pow(2)
        alpha_denom = 2 * grads.pow(2) + (acts * grads.pow(3)).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom
        weights = (alphas * F.relu(grads)).sum(dim=(2, 3))
        
        # Generate CAM
        cam = (weights[:, :, None, None] * acts).sum(1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        
        # Return 2D array
        return cam.squeeze(0).squeeze(0).cpu().numpy()

# Spatial Description Function
def generate_spatial_description(heatmap, threshold=0.6):
    # Convert heatmap to spatial region descriptions
    H, W = heatmap.shape
    
    # Find hot regions
    hot_mask = heatmap > threshold
    hot_pixels = np.argwhere(hot_mask)
    
    if len(hot_pixels) == 0:
        return "no strongly activated regions", 0.0
    
    y_coords, x_coords = hot_pixels[:, 0], hot_pixels[:, 1]
    y_center, x_center = int(y_coords.mean()), int(x_coords.mean())
    
    # Determine spatial location
    regions = []
    if y_center < H / 3:
        regions.append("top")
    elif y_center > 2 * H / 3:
        regions.append("bottom")
    else:
        regions.append("middle")
    
    if x_center < W / 3:
        regions.append("left")
    elif x_center > 2 * W / 3:
        regions.append("right")
    else:
        regions.append("center")
    
    region_desc = "-".join(regions)
    
    # Calculate coverage
    coverage = hot_mask.sum() / (H * W) * 100
    
    return region_desc, coverage

# Generate Text Explanation Function
def generate_text_explanation(image_tensor, heatmap, true_label, pred_label, pred_confidence, sample_idx):
    
    # Spatial analysis
    region_desc, coverage = generate_spatial_description(heatmap, threshold=0.6)
    
    # Activation statistics
    max_activation = float(heatmap.max())
    mean_activation = float(heatmap.mean())
    
    # Generate natural language explanation
    explanation_parts = []
    
    # 1. Prediction info
    explanation_parts.append(f"Image #{sample_idx}: The model classified this image as '{pred_label}' with {pred_confidence:.1f}% confidence.")
    
    if pred_label == true_label:
        explanation_parts.append(f"This prediction is CORRECT (ground truth: {true_label}).")
    else:
        explanation_parts.append(f"This prediction is INCORRECT (ground truth: {true_label}).")
    
    # 2. Spatial focus
    if coverage > 0:
        explanation_parts.append(f"The model focused primarily on the {region_desc} region of the image, with strong activations covering approximately {coverage:.1f}% of the image area.")
    else:
        explanation_parts.append("The model showed weak or diffuse attention across the image without clear focal regions.")
    
    # 3. Activation strength
    if max_activation > 0.8:
        explanation_parts.append(f"The maximum activation strength was very high ({max_activation:.2f}), indicating strong feature detection in key areas.")
    elif max_activation > 0.5:
        explanation_parts.append(f"The activation strength was moderate ({max_activation:.2f}), suggesting reasonable feature detection.")
    else:
        explanation_parts.append(f"The activation strength was relatively weak ({max_activation:.2f}), indicating uncertain feature detection.")
    
    # 4. Decision reasoning
    explanation_parts.append(f"These activations suggest the model identified visual patterns characteristic of '{pred_label}' class in these regions.")
    
    return " ".join(explanation_parts)

def gradcam_to_structured_json(sample_idx, img, true_label, pred_label, pred_confidence, heatmap, top_n_regions=3, threshold_ratio=0.6):
    """
    Convert a Grad-CAM++ heatmap into a structured JSON explanation for LLM input.
    
    Args:
        sample_idx (int): Index of the image.
        img (Tensor or np.array): Original image (for visualization, optional).
        true_label (str): Ground truth label.
        pred_label (str): Model prediction.
        pred_confidence (float): Prediction confidence (0-100).
        heatmap (np.array): 2D heatmap from Grad-CAM++.
        top_n_regions (int): Number of top regions to include.
        threshold_ratio (float): Fraction of max activation to consider as hotspot.
        
    Returns:
        dict: Structured JSON explanation.
    """
    
    # Global activation stats
    max_act = float(heatmap.max())
    mean_act = float(heatmap.mean())
    coverage_percent = float((heatmap > threshold_ratio * max_act).sum() / heatmap.size * 100)
    
    # Multi-region extraction
    threshold = threshold_ratio * max_act
    hot_mask = heatmap > threshold
    
    labeled_array, num_features = scipy_label(hot_mask)  # Use renamed import
    regions = []
    
    for region_idx in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == region_idx)
        if coords.size == 0:
            continue
        
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        y_center, x_center = int(y_coords.mean()), int(x_coords.mean())
        
        # Determine region name
        H, W = heatmap.shape
        vertical = "top" if y_center < H/3 else "bottom" if y_center > 2*H/3 else "middle"
        horizontal = "left" if x_center < W/3 else "right" if x_center > 2*W/3 else "center"
        region_name = f"{vertical}-{horizontal}"
        
        # Region stats
        region_mask = (labeled_array == region_idx)
        region_max = float(heatmap[region_mask].max())
        region_mean = float(heatmap[region_mask].mean())
        region_coverage = float(region_mask.sum() / heatmap.size * 100)
        
        regions.append({
            "region_name": region_name,
            "coverage_percent": region_coverage,
            "max_activation": region_max,
            "mean_activation": region_mean
        })
    
    # Sort regions by coverage or max activation
    regions = sorted(regions, key=lambda r: r["coverage_percent"], reverse=True)[:top_n_regions]
    
    # Textual explanation (for Stage 1 LLM)
    if pred_label == true_label:
        correctness = "correct"
        is_correct = True
    else:
        correctness = "incorrect"
        is_correct = False
        
    region_names_str = ", ".join([r["region_name"] for r in regions])
    
    textual_explanation = (
        f"Image #{sample_idx}: The model classified this image as '{pred_label}' "
        f"with {pred_confidence:.1f}% confidence. This prediction is {correctness} "
        f"(ground truth: {true_label}). The strongest activations were in {region_names_str} "
        f"regions, covering {coverage_percent:.1f}% of the image. The max activation was "
        f"{max_act:.2f} and mean activation {mean_act:.2f}, indicating feature detection "
        f"corresponding to the predicted class."
    )
    
    # Structured JSON
    explanation_json = {
        "sample_idx": sample_idx,
        "true_label": true_label,
        "predicted_label": pred_label,
        "prediction_confidence": float(pred_confidence),
        "is_correct": is_correct,
        "activations": {
            "max": max_act,
            "mean": mean_act,
            "coverage_percent": coverage_percent
        },
        "regions": regions,
        "textual_explanation": textual_explanation
    }
    
    return explanation_json

# plotting function
def plot_gradcam_publication(img_np, heatmap, true_label, pred_label, 
                             pred_confidence, sample_idx, save_path_base):

    # Resize heatmap to match image
    H, W = img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # Colorize heatmap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Overlay
    img_uint8 = np.uint8(255 * np.clip(img_np, 0, 1))
    overlay = cv2.addWeighted(
        img_uint8, 0.5,
        np.uint8(255 * heatmap_colored), 0.5,
        0
    ) / 255.0

    # 1. Save Original Image
    fig1 = plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    #plt.title(f"Ground Truth: {true_label}", fontsize=24, pad=12)
    plt.axis("off")

    plt.tight_layout()
    out1 = f"{save_path_base}" + f"CIFAR10_Original.png"
    plt.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. Save Grad-CAM++ Overlay
    fig2 = plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    # plt.title(
    #     f"Grad-CAM++ — {pred_label}",
    #     fontsize=24, pad=12
    # )
    plt.axis("off")

    plt.tight_layout()
    out2 = f"{save_path_base}" + f"CIFAR10_GradCAMpp.png"
    plt.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved original image to {out1}")
    print(f"Saved Grad-CAM++ image to {out2}")


# Find a convolutional layer to target - used for Grad-CAM++
target_layer = None
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        target_layer = module
        target_layer_name = name

if target_layer is None:
    raise ValueError("Could not find a convolutional layer in the model")

# Apply Grad-CAM++
cam = GradCAMpp(model, target_layer)

# Sample index to explain
sample_idx = 2
img, label = testset[sample_idx]

# Convert label tensor to integer
label_idx = int(label)
input_tensor = img.unsqueeze(0).to(device)

model.eval()
output = model(input_tensor)
probs = F.softmax(output, dim=1)
pred_idx = output.argmax(dim=1).item()
pred_confidence = probs[0, pred_idx].item() * 100

# Backward for Grad-CAM++
model.zero_grad()
one_hot = torch.zeros_like(output)
one_hot[0, pred_idx] = 1
output.backward(gradient=one_hot, retain_graph=True)

heatmap = cam.generate(pred_idx)

img_np = img.numpy().transpose(1, 2, 0)
img_np = img_np * np.array([0.2470, 0.2435, 0.2616]) + np.array([0.4914, 0.4822, 0.4465])
img_np = np.clip(img_np, 0, 1)

# Resize to 32x32 if needed (CIFAR-10)
if img_np.shape[0] != 32:
    img_np = cv2.resize(img_np, (32, 32))

# Generate figure
plot_gradcam_publication(
    img_np=img_np,
    heatmap=heatmap,  # generated heatmap
    true_label=classes[label_idx],
    pred_label=classes[pred_idx],
    pred_confidence=pred_confidence,
    sample_idx=sample_idx,
    save_path_base=OUTPUT_PATH,
)

# Verify shape
print(f"Heatmap shape: {heatmap.shape}")
assert heatmap.ndim == 2, f"Heatmap must be 2D, got shape {heatmap.shape}"

# Generate text explanation
text_explanation = generate_text_explanation(
    img, heatmap, classes[label_idx], classes[pred_idx], pred_confidence, sample_idx
)

# Full structured JSON explanation
structured_expl = gradcam_to_structured_json(
    sample_idx=sample_idx,
    img=img,
    true_label=classes[label_idx],
    pred_label=classes[pred_idx],
    pred_confidence=pred_confidence,
    heatmap=heatmap,
    top_n_regions=3,        # adjust if you want more regions
    threshold_ratio=0.6      # same as your previous threshold
)

json_path = OUTPUT_PATH + f"gradcampp_sample{sample_idx}_output.json"
with open(json_path, "w") as f:
    json.dump(structured_expl, f, indent=2)
print(f"Structured explanation saved to {json_path}")

# Prepare heatmap for visualization
heatmap_resized = cv2.resize(heatmap, (32, 32))                 # CIFAR10 is 32x32
heatmap_uint8 = np.uint8(255 * heatmap_resized)
heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

# Denormalize and prepare original image
img_np = img.numpy().transpose(1, 2, 0)
img_np = img_np * np.array([0.2470, 0.2435, 0.2616]) + np.array([0.4914, 0.4822, 0.4465])   # denormalize like we did at the start
img_np = np.clip(img_np, 0, 1)
img_uint8 = np.uint8(255 * img_np)

# Create overlay
overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_colored, 0.4, 0)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_np)
axes[0].set_title(f"Original\nTrue: {classes[label_idx]}")  # Fix: use label_idx
axes[0].axis("off")

axes[1].imshow(heatmap_resized, cmap='jet')
axes[1].set_title(f"Grad-CAM++\nPred: {classes[pred_idx]} ({pred_confidence:.1f}%)")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(DATA_PATH + "gradcampp_sample.png", dpi=150, bbox_inches='tight')
print(f"Visualization saved to {DATA_PATH}gradcampp_sample.png")