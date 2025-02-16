from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


def split_dataset_stratified(dataset, test_size=0.1):

    train_val, test, stratify_train_val, stratify_test = train_test_split(
        dataset, dataset.y, test_size=test_size, random_state=42, stratify=dataset.y
    )

    val_size_adjusted = test_size / (
        1 - test_size
    )  # creo que sera val_size / (1 - test_size)

    train, val, _, _ = train_test_split(
        train_val,
        stratify_train_val,
        test_size=val_size_adjusted,
        random_state=42,
        stratify=stratify_train_val,
    )

    return train, val, test


def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=16):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_dataset != None:
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
    else:
        val_loader = None
    if test_dataset != None:
        test_loader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=True
        )
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


def get_data_loaders_alt(train_dataset, val_dataset, test_dataset, batch_size=16):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_dataset != None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        val_loader = None
    if test_dataset != None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader
