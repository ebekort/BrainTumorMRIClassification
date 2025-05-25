#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:31:44 2025

@author: sennehollard
"""

# baseline_model_svm.py

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from tqdm import tqdm
import os
import torch
from project_name.utils.loader import get_dataloaders
import matplotlib.pyplot as plt

# --- Helper Functions ---

def extract_svm_features(loader, desc_text="Extracting features..."):
    """
    Extracts features and labels from a DataLoader.
    """
    X_list = []
    y_list = []
    print(desc_text) # tqdm might be better inside the loop or handled by caller
    for imgs_batch, labels_batch in tqdm(loader, desc=desc_text.split("...")[0]): # Use part of desc_text for tqdm
        labels_indices = torch.argmax(labels_batch, dim=1).tolist()
        imgs_flat = imgs_batch.view(imgs_batch.size(0), -1).numpy().tolist()
        X_list.extend(imgs_flat)
        y_list.extend(labels_indices)
    return np.array(X_list), np.array(y_list, dtype=int)


def plot_svm_coefficients(svm_model, classes, output_dir, image_shape=(224, 224)):
    """
    Visualizes feature importances for a linear SVM model.
    Saves plots to the specified output directory.
    """
    if svm_model.kernel == 'linear':
        print("Linear kernel was selected. Visualizing feature importances...")
        coefficients = svm_model.coef_
        
        num_features = coefficients.shape[1]
        expected_total_features = image_shape[0] * image_shape[1]
        if num_features != expected_total_features:
            print(f"Warning: Number of features ({num_features}) does not match expected image dimensions {image_shape} (total {expected_total_features}). Visualization might be incorrect.")
            # Proceeding anyway, but this check is important

        if coefficients.shape[0] == 1: # Binary classification or decision_function_shape='ovo' with 2 classes
            coef_img = coefficients[0].reshape(image_shape)
            class_name_for_plot = f"{classes[1]}_vs_{classes[0]}" if len(classes) == 2 else "binary_class_coef"
            
            plt.figure(figsize=(8, 8))
            plt.imshow(coef_img, cmap='viridis')
            plt.colorbar()
            plt.title(f"Feature importances for {class_name_for_plot}")
            filename = f"svm_linear_coef_{class_name_for_plot.replace(' ', '_')}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path)
            plt.close()
            print(f"Saved feature importance plot to {save_path}")
        else: # Multi-class classification (OVR)
            for i in range(coefficients.shape[0]):
                if i < len(classes):
                    class_name = classes[i]
                    coef_img = coefficients[i].reshape(image_shape)
                    
                    plt.figure(figsize=(8, 8))
                    plt.imshow(coef_img, cmap='viridis')
                    plt.colorbar()
                    plt.title(f"Feature importances for class: {class_name}")
                    filename = f"svm_linear_coef_class_{class_name.replace(' ', '_')}.png"
                    save_path = os.path.join(output_dir, filename)
                    plt.savefig(save_path)
                    plt.close()
                    print(f"Saved feature importance plot for class {class_name} to {save_path}")
                else:
                    print(f"Warning: More coefficient sets ({coefficients.shape[0]}) than class names ({len(classes)}). Skipping visualization for coefficient set {i}.")
        
        print(f"Feature importance visualization finished. Plots saved in {output_dir}")
    else:
        print(f"Non-linear kernel ({svm_model.kernel}) was selected. For explainability, consider using LIME or SHAP. Direct coefficient visualization is for linear kernels.")


# --- Main Script Logic ---

if __name__ == '__main__': # Protect main logic when script is imported
    
    # Define output directory for plots
    output_plot_dir = os.path.join("outputs", "svm_explainability")
    os.makedirs(output_plot_dir, exist_ok=True)

    # Define data directory and classes
    data_dir = "./data" # Path relative to the project root or where the script is run
    if not os.path.isdir(data_dir) and "models" in os.path.abspath(__file__):
        data_dir_alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        if os.path.isdir(data_dir_alt):
            data_dir = data_dir_alt
        else:
            data_dir_proj_name_data = os.path.join("project_name", "data")
            if os.path.isdir(data_dir_proj_name_data):
                data_dir = data_dir_proj_name_data
            else:
                raise FileNotFoundError(
                    f"Data directory not found. Checked: './data', '{data_dir_alt}', and '{data_dir_proj_name_data}'. "
                    f"Please ensure the path '{data_dir}' is correct relative to your execution directory or that the script is in 'project_name/models'."
                )
    elif not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please ensure the path is correct.")

    classes = [cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls)) and not cls.startswith('.')]
    if not classes:
        raise ValueError(f"No class subdirectories found in {data_dir}. Check the directory structure.")
    classes.sort() 

    num_avail_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    loader_num_workers = max(1, num_avail_workers // 2 if num_avail_workers else 1)

    train_loader, _, test_loader = get_dataloaders(data_dir, classes, batch_size=64, num_workers=loader_num_workers)

    # Extract features using helper function
    X_train, y_train = extract_svm_features(train_loader, desc_text="Extracting train features for SVM...")
    X_test, y_test = extract_svm_features(test_loader, desc_text="Extracting test features for SVM...")

    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define parameter grid for SVM
    param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1]}
    ]

    # Instantiate SVC for GridSearchCV
    svm_base = SVC(random_state=42) 

    # Instantiate GridSearchCV
    print("Starting GridSearchCV for SVM...")
    grid_search = GridSearchCV(estimator=svm_base, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best estimator and print best parameters
    best_svm = grid_search.best_estimator_
    print(f"Best SVM parameters: {grid_search.best_params_}")

    # Evaluate using the best SVM model
    print("Evaluating SVM with best parameters...")
    y_pred = best_svm.predict(X_test)

    # Plot coefficients if linear kernel, using helper function
    plot_svm_coefficients(best_svm, classes, output_plot_dir, image_shape=(224, 224))

    # Classification Report
    print("Classification Report:")
    report_labels = sorted(np.unique(np.concatenate((y_test, y_pred))))
    target_names_for_report = [classes[i] if i < len(classes) else f"Label {i}" for i in report_labels]
    print(classification_report(y_test, y_pred, labels=report_labels, target_names=target_names_for_report))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("SVM baseline model execution finished.")
