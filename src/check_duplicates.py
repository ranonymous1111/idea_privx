import torch
import argparse
import os


def check_duplicates(dataset_name="Cora", data_dir="../data"):
    """Check for duplicate/similar subgraphs in the dataset."""

    train_path = os.path.join(data_dir, f"{dataset_name}_train_data.pt")
    test_path = os.path.join(data_dir, f"{dataset_name}_test_data.pt")

    # Load both datasets
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Train or test data not found")
        return

    train_data = torch.load(train_path)
    test_data = torch.load(test_path)

    train_adj_list = [d["adj"] for d in train_data]
    test_adj_list = [d["adj"] for d in test_data]

    n_train = len(train_adj_list)
    n_test = len(test_adj_list)

    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 70}")
    print(f"Train samples: {n_train}")
    print(f"Test samples: {n_test}")

    # ============ Check within Train set ============
    print(f"\n--- Within TRAIN Set ---")
    train_hashes = [hash(adj.numpy().tobytes()) for adj in train_adj_list]
    unique_train = len(set(train_hashes))
    print(f"Unique samples: {unique_train}")
    print(
        f"Exact duplicates: {n_train - unique_train} ({100 * (n_train - unique_train) / n_train:.2f}%)"
    )

    # ============ Check within Test set ============
    print(f"\n--- Within TEST Set ---")
    test_hashes = [hash(adj.numpy().tobytes()) for adj in test_adj_list]
    unique_test = len(set(test_hashes))
    print(f"Unique samples: {unique_test}")
    print(
        f"Exact duplicates: {n_test - unique_test} ({100 * (n_test - unique_test) / n_test:.2f}%)"
    )

    # ============ Check TRAIN vs TEST overlap (exact matches) ============
    print(f"\n--- TRAIN vs TEST Overlap ---")
    train_hash_set = set(train_hashes)
    test_hash_set = set(test_hashes)
    overlap = train_hash_set & test_hash_set
    print(f"Exact same graphs in both train & test: {len(overlap)}")
    if len(overlap) > 0:
        # Count how many test samples are duplicates of train
        test_in_train = sum(1 for h in test_hashes if h in train_hash_set)
        print(
            f"Test samples that appear in train: {test_in_train} ({100 * test_in_train / n_test:.2f}% of test)"
        )

    # ============ Check similarity between TRAIN and TEST ============
    print(f"\n--- TRAIN vs TEST Similarity (Jaccard) ---")
    similarities = []
    num_comparisons = min(2000, n_train * n_test)

    for _ in range(num_comparisons):
        i = torch.randint(0, n_train, (1,)).item()
        j = torch.randint(0, n_test, (1,)).item()

        adj_train = train_adj_list[i].float()
        adj_test = test_adj_list[j].float()

        # Jaccard similarity of edges
        intersection = (adj_train * adj_test).sum()
        union = ((adj_train + adj_test) > 0).float().sum()
        if union > 0:
            similarities.append((intersection / union).item())

    if similarities:
        print(f"Random pairs compared: {len(similarities)}")
        print(f"  Mean Jaccard: {sum(similarities) / len(similarities):.4f}")
        print(f"  Max Jaccard: {max(similarities):.4f}")
        print(f"  Min Jaccard: {min(similarities):.4f}")

        # Distribution
        very_high = sum(1 for s in similarities if s > 0.9)
        high_sim = sum(1 for s in similarities if s > 0.5)
        medium_sim = sum(1 for s in similarities if 0.2 < s <= 0.5)
        low_sim = sum(1 for s in similarities if s <= 0.2)

        print(f"\n  Distribution:")
        print(
            f"    >90% similar: {very_high} ({100 * very_high / len(similarities):.2f}%)"
        )
        print(
            f"    >50% similar: {high_sim} ({100 * high_sim / len(similarities):.2f}%)"
        )
        print(
            f"    20-50% similar: {medium_sim} ({100 * medium_sim / len(similarities):.2f}%)"
        )
        print(
            f"    <=20% similar: {low_sim} ({100 * low_sim / len(similarities):.2f}%)"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed"]
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    args = parser.parse_args()

    check_duplicates(args.dataset, args.data_dir)


if __name__ == "__main__":
    main()
