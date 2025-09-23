import torch
import torch.nn as nn
from selectga.utils.utils import compute_metrics, setup_logger
from pathlib import Path
from selectga.models.ga_predictor import GAEstimator
from selectga.data.dataset_sweep import FocusSweepDataset, FocusSweepDataset_
from transforms import USAugmentation
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import argparse
from tqdm import tqdm
import traceback
from torch.amp import autocast, GradScaler


scaler = GradScaler() 

def train_epoch(train_loader,model, device,optimizer, criterion, logger=None):
    model.train()
    train_loss = 0
    num_samples = 0
    for frames, labels, _ in train_loader:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast("cuda"):  # Enable Mixed Precision
            predictions = model(frames)
            loss = criterion(predictions.squeeze(), labels.squeeze())

        scaler.scale(loss).backward()  
        scaler.step(optimizer)  
        scaler.update() 

        train_loss += (loss.item() * labels.size(0))
        num_samples += labels.size(0)

        del predictions, labels, frames, loss
        torch.cuda.empty_cache()

    avg_train_loss =  train_loss/ num_samples #len(train_loader)

    return avg_train_loss

def evaluate_epoch(val_loader, criterion, model, device, logger=None):
    model.eval()
    val_mae = 0
    val_loss = 0
    num_samples = 0
    study_ids = []
    # all_preds = []
    # all_labels = []

    with torch.no_grad():
        for frames, labels, _ in val_loader:
            
            frames = frames.to(device)
            labels = labels.to(device)
            predictions = model(frames)
            # all_preds.append(predictions.cpu().numpy())
            # all_labels.append(labels.cpu().numpy())
            loss  = criterion(predictions, labels)
            val_loss += (loss.item() * labels.size(0))
            val_mae += torch.abs(predictions - labels).sum().item()
            num_samples += labels.size(0)

            del predictions, labels, frames, loss
            torch.cuda.empty_cache()


    avg_val_loss = val_loss / num_samples
    # val_metrics = compute_metrics(predictions=all_preds, ground_truth=all_labels)
    val_mae /= num_samples

    return avg_val_loss, val_mae #, val_metrics

def train_fold(args, logger, device):
    '''
    train single fold with specified GPU
    '''
    
    model = GAEstimator().to(device)

    img_size = (224,224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = USAugmentation(img_size=img_size,
                                    mean=mean,
                                    std=std,
                                    mode= 'train',
                                    seed=42,
                                    padding=16)
    
    train_dataset = FocusSweepDataset_(args.train_file,
                                      sampling=args.sampling,
                                      transform=train_transform)
 
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=True)

    val_transform = USAugmentation(img_size=img_size,
                                    mean=mean,
                                    std=std,
                                    mode='val',
                                    seed=42,
                                    padding=16)
    val_dataset = FocusSweepDataset_(args.val_file,
                                    sampling=args.sampling,
                                    transform=val_transform)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=0.005)
    scheduler = StepLR(optimizer, step_size=45, gamma=0.1) 
    criterion = nn.L1Loss()

    best_mae = float('inf')
    patience_counter = 0

    model_save_path = Path(args.save_dir) /f'best_{args.model}_{args.sampling}_fold_{args.fold}.pth' # TODO: remove usfm if not using itd backbone

    logger.info(f"Starting training fold {args.fold}")
    train_losses = []
    val_losses = []
    try:
        for epoch in tqdm(range(args.max_epochs), "Training epoch: "):
            model.train()
            train_loss = train_epoch(train_loader,model, device, optimizer, criterion)
            train_losses.append(train_loss)
            logger.info(f"Epoch: {epoch}/{args.max_epochs:.0f} , Train loss: {train_loss:.4f}")
            if epoch % args.eval_frequency == 0:
                val_loss, val_mae = evaluate_epoch(val_loader,criterion, model, device)
                val_losses.append(val_loss)
                logger.info(f"Epoch: {epoch}/{args.max_epochs:.0f} , Train loss: {train_loss:.4f} , Val loss: {val_loss:.4f} ")
                # logger.info(f"Evaluation metrics at epoch {epoch}: {val_metrics}")
                if val_loss < best_mae:
                    best_mae = val_loss
                    patience_counter = 0

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_mae': val_loss
                    }, model_save_path)
                    logger.info(f"Saved new best model af epoch {epoch} with Mean Absolute Error: {val_loss:.4f} days) ")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        logger.info(f"Early stopping triggered after {epoch} epochs")
                        break
            scheduler.step()
    except Exception as e:
        logger.info(f"Error training on video {e} {traceback.format_exc()}")

    logger.info(f"Training completed for fold: {args.fold}")

    return model_save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, help='Fold number (0-4)')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data file')
    parser.add_argument('--val_file', type=str, required=True, help='Path to validation data file')
    parser.add_argument('--sampling', type=str,default='optimal',help='Sampling stategy for frame selection')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models and logs')
    parser.add_argument('--model',type=str, default='resnet',help='Tpe of model backbone. Avalilable options: "resnet" or "usfm".')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=300, help='Maximum number of epochs')
    parser.add_argument('--eval_frequency', type=int, default=5, help='Evaluation frequency')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()

    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.set_device(args.gpu_id)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(save_dir,args.model, args.fold, args.sampling)
    
    # Train fold
    save_path = train_fold(args, logger, device)
