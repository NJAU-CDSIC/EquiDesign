import argparse
import torch
import numpy as np
from datasets import ts50_dataset  # <-- 正确导入 ts50_dataset
from model_utils import EquiDesign, loss_nll


def evaluate_test_set(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading TS50 dataset...")
    test_set = ts50_dataset(batch_size=args.batch_size)  # ts50_dataset 返回的是 DynamicLoader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    print("Loading model checkpoint...")
    checkpoint = torch.load(args.previous_checkpoint, map_location=device, weights_only=False)

    model = EquiDesign(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_encoder_layers,
        k_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
        equiformer_out_vector=getattr(args, "equiformer_out_vector", 0),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    total_loss = 0
    total_acc = 0
    total_weights = 0

    with torch.no_grad():
        for batch in test_loader:
            X, S, mask, SS = batch
            X = X.squeeze(0).to(device)
            S = S.squeeze(0).long().to(device)
            mask = mask.squeeze(0).to(device)
            SS = SS.squeeze(0).to(device)
            chain_M = torch.ones_like(S).to(device)
            mask_for_loss = mask * chain_M

            log_probs = model(X, S, mask, chain_M, ss=SS)
            loss, _, true_false = loss_nll(S, log_probs, mask_for_loss)

            total_loss += torch.sum(loss * mask_for_loss).item()
            total_acc += torch.sum(true_false * mask_for_loss).item()
            total_weights += torch.sum(mask_for_loss).item()

    test_loss = total_loss / total_weights
    test_accuracy = total_acc / total_weights
    test_perplexity = np.exp(test_loss)

    print(f"[TS50 EVALUATION - EquiDesign]")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Perplexity: {test_perplexity:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model on the TS50 test set.")
    parser.add_argument("--path_for_outputs", type=str, required=True)
    parser.add_argument(
        "--previous_checkpoint",
        type=str,
        default="../Model/model_weights/best_model.pt",
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--num_neighbors", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--backbone_noise", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--equiformer_out_vector", type=int, default=0)

    args = parser.parse_args()
    evaluate_test_set(args)
