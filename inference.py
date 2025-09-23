from selectga.models.ga_predictor import GAEstimator
from selectga.utils.utils import compute_metrics, setup_logger, plot_evaluation_results
from selectga.data.dataset_sweep import FocusSweepDataset, FocusSweepDataset_
from selectga.data.transforms import USAugmentation
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from collections import defaultdict
import os
import csv


MODEL_PATH_BASE = "checkpoints"

def load_models(device, num_folds=5):
    models = []
    for i in range(num_folds):
        model_path = Path(MODEL_PATH_BASE) / f"fold_{i}/model_fold_{i}.pth"
        model = GAEstimator()
        chkpt = torch.load(model_path)
        model.load_state_dict(chkpt['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)

    return models
    

def infer(test_loader, models, device, save_dir,logger=None):
    # model.eval()
    val_mae = 0
    val_loss = 0
    num_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            model_preds = []
            for model in models:
                predictions = model(frames)
                model_preds.append(predictions)
            ensemble_preds = torch.mean(torch.stack(model_preds), dim=0)
            all_preds.append(ensemble_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())


            del predictions, model_preds, ensemble_preds, labels, frames
            torch.cuda.empty_cache()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    val_metrics = compute_metrics(predictions=all_preds, ground_truth=all_labels)
    logger.info(f"Metrics on inference: {val_metrics}")
    plot_evaluation_results(all_preds, all_labels, save_dir)
    
    logger.info(f"Finished inference. Plots saved in {save_dir}")

    return None

def infer_single_model(test_dataset, model, device, save_dir,logger=None, model_name='resnet-random'):
    # model.eval()
    val_mae = 0
    # Initialize dictionaries to store predictions and labels for each patient ID
    patient_preds = defaultdict(list)
    patient_labels = {}
    all_embeddings = []
    all_labels = []

    # Assuming test_dataset is your DataLoader
    with torch.no_grad():
        for frames, labels, patient_id in test_dataset:
            frames = frames.to(device).unsqueeze(0)  # add batch dimension
            labels = labels.to(device)
            
            # Get predictions from the model
            predictions, embeddings = model(frames)#
            embeddings = embeddings.squeeze()
            predictions = predictions.squeeze()
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            # Store predictions and labels in the dictionaries
            patient_preds[patient_id].append(predictions.cpu().numpy())
            patient_labels[patient_id] = labels.cpu().numpy()  # Ground truth is the same for all items with the same patient ID

            del predictions, labels, frames
            torch.cuda.empty_cache()

    # Average predictions for each patient ID
    averaged_preds = []
    final_labels = []
    patient_ids = []

    for patient_id in patient_preds:
        # Average predictions for the current patient ID
        avg_pred = np.mean(patient_preds[patient_id], axis=0)
        averaged_preds.append(avg_pred)
        
        # Append the corresponding ground truth label
        final_labels.append(patient_labels[patient_id])
        patient_ids.append(patient_id)

    # Convert lists to numpy arrays for easier metric computation
    all_preds = np.array(averaged_preds)
    # all_labels = np.array(final_labels)

    # Write results to CSV file
    csv_path = os.path.join(save_dir, f'{model_name}_test_predictions.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header row
        csv_writer.writerow(['patient_id', 'label', 'prediction', 'trimester', 'country'])
        
        # Write data rows
        for i in range(len(patient_ids)):
            # Determine trimester based on label value
            label_value = final_labels[i]
            if 97 <= label_value <= 195:
                trimester = "2nd"
            elif label_value > 195:
                trimester = "3rd"
            else:
                trimester = "1st"  # For values < 97
            
            # Determine country based on patient ID prefix
            patient_id = patient_ids[i]
            country = "Kenya" if patient_id.startswith("KE") else "Spain"
            
            csv_writer.writerow([
                patient_ids[i], 
                final_labels[i], 
                averaged_preds[i], 
                trimester, 
                country
            ])

    # all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
     # Compute all metrics using the CSV file
    val_metrics = compute_metrics(csv_path=csv_path)
    
    # Log overall metrics
    if logger:
        logger.info(f"Metrics on inference for {len(all_preds)} test patients:")
        logger.info(f"Overall metrics: MAE={val_metrics['overall']['mae']:.3f}±{val_metrics['overall']['mae_std']:.3f}, "
                   f"RMSE={val_metrics['overall']['rmse']:.3f}±{val_metrics['overall']['rmse_std']:.3f}, "
                   f"R²={val_metrics['overall']['r2']:.3f}, "
                   f"Within 7 days: {val_metrics['overall']['within_7_days']:.1f}%, "
                   f"Within 14 days: {val_metrics['overall']['within_14_days']:.1f}% ")
        
        # Log metrics by trimester
        logger.info("Metrics by trimester:")
        for trimester, metrics in val_metrics['by_trimester'].items():
            logger.info(f"  {trimester} trimester ({metrics['count']} patients): "
                       f"MAE={metrics['mae']:.3f}±{metrics['mae_std']:.3f}, "
                       f"RMSE={metrics['rmse']:.3f}±{metrics['rmse_std']:.3f}, "
                       f"R²={metrics['r2']:.3f}, "
                       f"Within 7 days: {metrics['within_7_days']:.1f}%, "
                       f"Within 14 days: {metrics['within_14_days']:.1f}%")
        
        # Log metrics by country
        logger.info("Metrics by country:")
        for country, metrics in val_metrics['by_country'].items():
            logger.info(f"  {country} ({metrics['count']} patients): "
                       f"MAE={metrics['mae']:.3f}±{metrics['mae_std']:.3f}, "
                       f"RMSE={metrics['rmse']:.3f}±{metrics['rmse_std']:.3f}, "
                       f"R²={metrics['r2']:.3f}, "
                       f"Within 7 days: {metrics['within_7_days']:.1f}% , "
                       f"Within 14 days: {metrics['within_14_days']:.1f}%")
    
        logger.info(f"Finished inference. Plots saved in {save_dir}")

    return None


if __name__ == "__main__":
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.set_device(args.gpu_id)

    # Create save directory
    save_dir = Path("/home/tanya-akumu/gestation_age/ResNet50_GA/results/embedings_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)
    model_type='resnet'
    
    # Setup logger
    logger = setup_logger(save_dir,model_type, 'inference', 'random')
    
    # run inference

    model = GAEstimator().to(device)
    chkpt = torch.load(Path("/checkpoints/ResNet50_GA/results/fold_0/best_resnet_model_fold_0.pth"))
    
    model.to(device)
       
    model.load_state_dict(chkpt['model_state_dict'])
    model.eval()

    img_size = (224,224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_file = "/dataset/data/optimal_frames_sweeps.csv"
    test_transform = USAugmentation(img_size=img_size,
                                    mean=mean,
                                    std=std,
                                    mode='test',
                                    padding=16)
    test_dataset = FocusSweepDataset_(test_file,
                                    # model='video',
                                    sampling='uniform',
                                    transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=False)
    infer_single_model(test_dataset, model, device, save_dir,logger, f"{model_type}-imagenet-init")
   
    