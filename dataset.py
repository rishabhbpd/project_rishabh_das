import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import resize_x, resize_y, batch_size


# ── Transforms ────────────────────────────────────────────────────────
def get_transforms():
    return transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset ───────────────────────────────────────────────────────────
class ForestPersonsDataset(datasets.ImageFolder):
    """
    Binary classification dataset loaded from local data/ directory.

    Expects the following structure:
        data/
        ├── train/
        │   ├── human/
        │   └── no_human/
        ├── val/
        │   ├── human/
        │   └── no_human/
        └── test/
            ├── human/
            └── no_human/

    Inherits from torchvision.datasets.ImageFolder which automatically
    assigns class labels based on subfolder names.
    Classes: {'human': 1, 'no_human': 0}  (alphabetical order)
    """

    def __init__(self, split="train", transform=None):
        """
        Args:
            split     : one of "train", "val", "test"
            transform : torchvision transforms (defaults to get_transforms())
        """
        root = os.path.join("data", split)
        if transform is None:
            transform = get_transforms()
        super().__init__(root=root, transform=transform)


# ── DataLoader ────────────────────────────────────────────────────────
def get_dataloader(split="train", shuffle=None):
    """
    Returns a DataLoader for the given split.

    Args:
        split   : "train", "val", or "test"
        shuffle : True for train, False otherwise (auto if None)
    """
    dataset = ForestPersonsDataset(split=split)

    if shuffle is None:
        shuffle = (split == "train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
    )
    return loader


# ── Sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing ForestPersonsDataset...")

    for split in ["train", "val", "test"]:
        dataset = ForestPersonsDataset(split=split)
        print(f"\n{split} split:")
        print(f"  Total samples : {len(dataset)}")
        print(f"  Classes       : {dataset.classes}")
        print(f"  Class indices : {dataset.class_to_idx}")

        sample, label = dataset[0]
        print(f"  Sample tensor : {sample.shape}  label={label}")

    print("\ndataset.py looks good!")
