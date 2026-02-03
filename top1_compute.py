import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import argparse
import pathlib

#Parse arguments from terminal
parser = argparse.ArgumentParser(description='Run method on files.')

parser.add_argument('--output_dir', type=str, help='Output directory')
parser.add_argument('--name', type=str, help='Dataset name')
parser.add_argument('--preprocessing.meta', type=str, help='Preprocessed metadata')
parser.add_argument('--ScoringTools.fullNES', type=str, help='Tools output collector')
parser.add_argument('--input_type', type=str, help='Tools of origin for input to the script for metrics computation')

args, _ = parser.parse_known_args()

dataset_name = getattr(args, 'name')
input_filepath = getattr(args, 'ScoringTools.fullNES')
output_dir = getattr(args, 'output_dir')
metadata = getattr(args, "preprocessing.meta")

os.makedirs(f"./{output_dir}", exist_ok = True)

#Load and preprocess metadata
metadata_df = pd.read_csv(metadata, sep = '\t')
metadata_df["true_label"] = metadata_df["true_label"].str.upper()

#Make list of all label types collected from metadata (samples taken from different label groups (e.g. tissues, sex, cell cycle phases etc.))
label_types_samples = list(set(metadata_df["true_label"]))
label_types_samples = [label_class.upper() for label_class in label_types_samples if isinstance(label_class, str)]

#Load in parameters dict (connect hash value to tool name).
parameters_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(input_filepath)), "parameters_dict.tsv"), delimiter="\t")
parameters_connect_dict = dict()
methods_connect_dict = dict()

for i, row in parameters_df.iterrows():
    hash_value = row["base_path"].split("/")[-1]
    human_value = row["alias_path"].split("-")[-1]
    parameters_connect_dict[hash_value] = human_value
    human_value = row["alias_path"].split("-")[1].split("_")[0]
    methods_connect_dict[hash_value] = human_value

#read data from tool
filepath_split = input_filepath.split("/")
module = filepath_split[-3]
parameter = filepath_split[-2]

if module == 'gsvaalone':
    module = methods_connect_dict[parameter]

if parameter != "default":
    parameter = parameters_connect_dict[parameter]

tool = module + "/" + parameter
tool_ = module + "_" + parameter

print(tool)

def process_NES_data():

    NES_data_df = pd.read_csv(input_filepath, sep = '\t')
    NES_data_df["GOI_Set"] = NES_data_df['GOI_Set'].str.upper() #Ensure that all gene sets (in GOI_Set) are in uppercase to match metadata true label 

    #List of gene signatures corresponding to different label classes with tool
    gene_signatures = list(NES_data_df["GOI_Set"])
    print("Gene sets tested: ", gene_signatures)

    #Get a list of all label classes present both in samples and having corresponding gene signatures
    geneset_types_shared = set(label_types_samples).intersection(set(gene_signatures))
    metadata_samples_signatures = metadata_df[metadata_df['true_label'].str.upper().isin(geneset_types_shared)]

    #Initialize
    samples_gt_and_preds_dict = {}

    #Add to dict only samples where the true label is a gene signature 
    for i, row in metadata_samples_signatures.iterrows():
        samples_gt_and_preds_dict[row['filename']] = {}
        samples_gt_and_preds_dict[row['filename']]['true_label'] = row['true_label']

    #Loop over each tool dataframe and get highest scoring gene set (highest NES) for each sample
    #Set index to GOI_Set
    NES_data_df = NES_data_df.set_index("GOI_Set")

    for i, row in NES_data_df.T.iterrows():
        sample_name = row.name
        if sample_name in samples_gt_and_preds_dict.keys():
            highest_tissue = row.idxmax(skipna=True)
            samples_gt_and_preds_dict[sample_name]["predicted_label"] = highest_tissue

    # Keep only samples that have predictions
    samples_gt_and_preds_dict = {
        k: v for k, v in samples_gt_and_preds_dict.items() 
        if "predicted_label" in v
    }

    print(f"Samples with both ground truth and predictions: {len(samples_gt_and_preds_dict)}")


    #df_preds contain samples as rows, coliumns are the true label and predicted labels with each tool
    df_preds = pd.DataFrame(samples_gt_and_preds_dict).T

    return df_preds, NES_data_df

def report_nas(output_dir, tool_, NES_data_df, dataset_name):
    ###Report NAs for tool, dataset combination###
    #Check for missing values in each tools' NES df and write information to file
    infile_nas = open(f"./{output_dir}/{dataset_name}-NA_data.tsv", "w")
    infile_nas.write(f"tool\tpercentage_NA_values\tpercentage_samples_with_NAs\n")

    #1. Get percent of all values that are NA
    total_values = NES_data_df.size
    total_na = NES_data_df.isna().sum().sum()
    pct_na_values = total_na / total_values

    #2. get percent of samples (columns) with at least one NA
    cols_with_na = NES_data_df.isna().any().sum()
    total_cols = NES_data_df.shape[1]
    pct_cols_with_na = cols_with_na / total_cols

    infile_nas.write(f"{tool_}\t{round(pct_na_values, 4)}\t{round(pct_cols_with_na, 4)}\n")
    infile_nas.close()


def filter_preds(y_pred, y_true):
    y_pred = np.array(y_pred, dtype=object)
    y_true = np.array(y_true, dtype=object)

    # Boolean mask: True where y_pred is NOT NaN
    mask = ~pd.isna(y_pred)  # or np.isnan for numeric arrays

    # Apply mask
    y_pred_filt = y_pred[mask]
    y_true_filt = y_true[mask]

    # If empty, return False
    if len(y_pred_filt) == 0:
        return False

    return list(y_pred_filt), list(y_true_filt)

def calculate_metrics(df_preds, tool_, dataset_name):

    y_pred = list(df_preds["predicted_label"])
    y_true = list(df_preds["true_label"])
    true_label_distributions = df_preds["true_label"].value_counts()
    labels = sorted(df_preds["true_label"].unique())

    # add filtering to check if all values are NA, then no metrics possible to be computed
    filter_output = filter_preds(y_pred, y_true)
    if filter_output is False:
        metrics_df = pd.DataFrame({
            'Tool': tool_,
            'MCC': [np.nan],
            'F1_Macro': [np.nan],
            'F1_Weighted': [np.nan],
            'Precision_Weighted': [np.nan],
            'Recall_Weighted': [np.nan]
        })
        metrics_df.to_csv(f'{output_dir}/{dataset_name}-top1metrics.tsv', sep='\t', index=False)
        pathlib.Path(f'{output_dir}/{dataset_name}-top1metrics-label_classes.tsv').touch()
        return
    else:
        y_pred, y_true = filter_preds(y_pred, y_true)
        labels = sorted(list(set(y_true))) # because otherwise will fail if removed due to na presence (all scores are NA)

    #If we only have one label class (e.g. in the hypoxia test sets)
    if len(list(set(y_true))) == 1:
        
        assert len(labels) == 1
        
        recall = recall_score(y_true, y_pred, pos_label=labels[0], average="weighted")
            
        #Create a df with tools as rows and metrics as columns
        metrics_df = pd.DataFrame({
            'Tool': tool_,
            'MCC': [np.nan],
            'F1_Macro': [np.nan],
            'F1_Weighted': [np.nan],
            'Precision_Weighted': [np.nan],
            'Recall_Weighted': round(recall, 4)
        })
    
        #Write to tsv file directly in output dir
        metrics_df.to_csv(f'{output_dir}/{dataset_name}-top1metrics.tsv', sep='\t', index=False)
    
        # --- Save per-label-class metrics ---
        # Create rows for each label
        label_metrics_rows = []
        for label in labels:
            label_metrics_rows.append({
                'Label_Class': label,
                'Count': true_label_distributions.get(label, 0),  # Add count
                'F1': np.nan,
                'Precision': np.nan,
                'Recall': round(recall, 6)
            })
        
        label_metrics_df = pd.DataFrame(label_metrics_rows)
        label_metrics_df.to_csv(f'{output_dir}/{dataset_name}-top1metrics-label_classes.tsv', sep='\t', index=False)
        
        
    else:
        #Initialize
        mcc_dict = {}
        f1_dict_macro = {}
        f1_dict_weighted = {}
        precision_dict_weighted = {} 
        recall_dict_weighted = {}    
    
        #Validation performance on individual label types 
        f1_gene_signatures_dict = {}
        precision_gene_signatures_dict = {}
        recall_gene_signatures_dict = {}
    
        #Extract true label, predicted label and list of possible labels, and true label distribution (how many samples in each true label class)
        
        #calculate performance metrics and confusion matrix
        # --- overall MCC ---
        mcc = round(matthews_corrcoef(y_true, y_pred), 3)
        mcc_dict[tool_] = mcc
        
        # --- overall F1; macro average and weighted average ---
        f1_macro = round(f1_score(y_true, y_pred, average="macro"), 3)
        f1_dict_macro[tool_] = f1_macro 
        
        f1_weighted = round(f1_score(y_true, y_pred, average="weighted"), 3)
        f1_dict_weighted[tool_] = f1_weighted 
        
        # --- overall precision and recall ---
        precision_weighted = round(precision_score(y_true, y_pred, average="weighted"), 3)
        precision_dict_weighted[tool_] = precision_weighted 
        
        recall_weighted = round(recall_score(y_true, y_pred, average="weighted"), 3)
        recall_dict_weighted[tool_] = recall_weighted 
        
        # --- per-class precision, recall and F1 ---
        metrics_report_per_tissue = classification_report(y_true, y_pred, output_dict=True)
        
        #initialize nested dicts for metric for each label group 
        f1_gene_signatures_dict[tool_] = {}  
        precision_gene_signatures_dict[tool_] = {}  
        recall_gene_signatures_dict[tool_] = {}  
        
        #iterate over each label group / gene set 
        for label in labels:
            # --- f1 ---
            f1 = metrics_report_per_tissue[label]["f1-score"]
            f1_gene_signatures_dict[tool_][label] = f1  
            
            # --- precision ---
            precision = metrics_report_per_tissue[label]["precision"]
            precision_gene_signatures_dict[tool_][label] = precision  
            
            # --- recall ---
            recall = metrics_report_per_tissue[label]["recall"]
            recall_gene_signatures_dict[tool_][label] = recall  
    
    
        #Create a df with tools as rows and metrics as columns
        metrics_df = pd.DataFrame({
            'Tool': list(mcc_dict.keys()),
            'MCC': [float(v) for v in mcc_dict.values()],
            'F1_Macro': [float(v) for v in f1_dict_macro.values()],
            'F1_Weighted': [float(v) for v in f1_dict_weighted.values()],
            'Precision_Weighted': [float(v) for v in precision_dict_weighted.values()],
            'Recall_Weighted': [float(v) for v in recall_dict_weighted.values()]
        })
    
        #Write to tsv file directly in output dir
        metrics_df.to_csv(f'{output_dir}/{dataset_name}-top1metrics.tsv', sep='\t', index=False)
    
        # --- Save per-label-class metrics ---
        # Create rows for each label
        label_metrics_rows = []
        for label in labels:
            label_metrics_rows.append({
                'Label_Class': label,
                'Count': true_label_distributions.get(label, 0),  # Add count
                'F1': f1_gene_signatures_dict[tool_][label],
                'Precision': precision_gene_signatures_dict[tool_][label],
                'Recall': recall_gene_signatures_dict[tool_][label]
            })
        
        label_metrics_df = pd.DataFrame(label_metrics_rows)
        label_metrics_df.to_csv(f'{output_dir}/{dataset_name}-top1metrics-label_classes.tsv', sep='\t', index=False)
    
    
        #confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  #row-normalized (true-label wise)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
        #Absolute counts
        ax = axes[0]
        im = ax.imshow(cm, cmap='Greens')
        ax.set_title(f"{tool} - Counts")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([label.capitalize() for label in labels], rotation=45, ha='right')
        ax.set_yticklabels([label.capitalize() for label in labels])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha='center', va='center')
        fig.colorbar(im, ax=ax)
    
        #Normalized
        ax = axes[1]
        im = ax.imshow(cm_norm, cmap='Greens', vmin=0, vmax=1)
        ax.set_title(f"{tool} - Normalized")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([label.capitalize() for label in labels], rotation=45, ha='right')
        ax.set_yticklabels([label.capitalize() for label in labels])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center')
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f"./{output_dir}/{dataset_name}-confusion_matrix_{tool_}", dpi=300, bbox_inches='tight')
        plt.close()


df_preds, NES_data_df = process_NES_data()

#Generate outputs: calculate_metrics and report NAs
calculate_metrics(df_preds, tool_, dataset_name)
report_nas(output_dir, tool_, NES_data_df, dataset_name)

pathlib.Path(f"{output_dir}/{dataset_name}-metrics_complete.flag").touch()
