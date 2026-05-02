import torch
from PIL import Image
from torchvision import transforms
from config import resize_x, resize_y, checkpoint_path
from model import ForestPersonCNN


# ── Class names (ImageFolder assigns alphabetically) ──────────────────
CLASS_NAMES = {0: "human", 1: "no_human"}


def get_model(device):
    """Load the trained model from checkpoint."""
    model = ForestPersonCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def predict(list_of_img_paths):
    """
    Run inference on a list of image file paths.

    Args:
        list_of_img_paths (list[str]): paths to .jpg image files

    Returns:
        list[str]: predicted class for each image
                   ("human" or "no_human")
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = get_model(device)
    transform = get_transform()

    results = []

    for img_path in list_of_img_paths:
        image  = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)  # add batch dim

        with torch.no_grad():
            output = model(tensor)
            pred   = output.argmax(dim=1).item()

        results.append(CLASS_NAMES[pred])

    return results


# ── Sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    # Grab a few images from data/test/
    human_dir    = os.path.join("data", "test", "human")
    no_human_dir = os.path.join("data", "test", "no_human")

    test_paths = (
        [os.path.join(human_dir, f)    for f in os.listdir(human_dir)[:3]] +
        [os.path.join(no_human_dir, f) for f in os.listdir(no_human_dir)[:3]]
    )

    print("Running predict on test images...")
    predictions = predict(test_paths)

    for path, pred in zip(test_paths, predictions):
        true_label = "human" if "no_human" not in path else "no_human"
        correct    = "✓" if pred == true_label else "✗"
        print(f"  {correct} {os.path.basename(path):20s} → predicted: {pred}")

    print("\npredict.py looks good!")
