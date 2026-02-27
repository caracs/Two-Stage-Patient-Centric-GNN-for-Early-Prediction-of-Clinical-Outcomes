# tesi_patient_fixed.py
import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix

import matplotlib.pyplot as plt


from torch_geometric.data import Data as GeoData
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, JumpingKnowledge

# -------------------------
# --- Config / Columns ---
# -------------------------
DATA_PATH = 'dataset_clean.csv'

df = pd.read_csv(DATA_PATH)
delay_max = df['hosp_to_icu_1'].max() if not df['hosp_to_icu_1'].isnull().all() else 1.0
dur_max   = df['icu_duration_1'].max() if not df['icu_duration_1'].isnull().all() else 1.0

vital_cols  = [ 'mean_hr','min_hr','max_hr','mean_sysbp','min_sysbp','max_sysbp',
                'mean_rr','min_rr','max_rr','mean_temp','min_temp','max_temp',
                'mean_spo2','min_spo2','max_spo2','mean_glucose','min_glucose','max_glucose' ]
vitals_mean = df[vital_cols].mean()
vitals_std  = df[vital_cols].std().replace(0,1)
vitals_fill = vitals_mean.to_dict()

cat_cols = ['gender','age_group','adm_type','adm_loc',
            'bicarbonate_test','creatinine_test','glucose_test','ast_test','bilirubin_test','hematocrit_test']

# -------------------------
# --- Stage1: Transformer (admission-level) ---
# -------------------------
class ClinicalStateEstimator(nn.Module):
    def __init__(self, n_categories, cat_emb_dim=32, attn_heads=8, attn_layers=2,
                 mlp_hidden_dims=[128,64], latent_dim=32, dropout=0.2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(num_cat, cat_emb_dim) for num_cat in n_categories])
        self.cls_token = nn.Parameter(torch.zeros(1,1,cat_emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=cat_emb_dim, nhead=attn_heads,
                                                   dim_feedforward=cat_emb_dim*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)
        layers=[]
        dims=[cat_emb_dim] + mlp_hidden_dims
        for a,b in zip(dims,dims[1:]):
            layers += [nn.Linear(a,b), nn.LayerNorm(b), nn.ReLU(), nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*layers)
        self.fc_latent = nn.Linear(mlp_hidden_dims[-1], latent_dim)
        self.heads = nn.ModuleDict({
            'icu_risk': nn.Linear(latent_dim, 1),
            'delay':    nn.Linear(latent_dim, 1)
        })

    def forward(self, x_cat):
        # x_cat: [B, n_cat]
        bsz = x_cat.size(0)
        cat_toks = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        cat_toks = torch.stack(cat_toks, dim=1)            # [B, n_cat, emb]
        cls = self.cls_token.expand(bsz, -1, -1)
        tokens = torch.cat([cls, cat_toks], dim=1)         # [B, seq_len, emb]
        attn_out = self.transformer(tokens)                # [B, seq_len, emb]
        pooled = attn_out[:, 0, :]                         # [B, emb]
        h = self.encoder(pooled)
        z = F.relu(self.fc_latent(h))                      # [B, latent_dim]
        out = {'latent': z, 'icu_risk': self.heads['icu_risk'](z).squeeze(-1), 'delay': self.heads['delay'](z).squeeze(-1)}
        return out

# -------------------------
# --- Admission Dataset (Stage1 training) ---
# -------------------------
class AdmissionDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.cat_cols = cat_cols
        self.df['icu_duration_1'] = self.df['icu_duration_1'].fillna(0.0).astype(float)
        self.df['was_in_icu'] = self.df['was_in_icu'].fillna(0).astype(int)
        self.df['hosp_to_icu_1'] = self.df['hosp_to_icu_1'].fillna(0.0).astype(float)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cat_vals = [int(row[c]) for c in self.cat_cols]
        x_cat = torch.tensor(cat_vals, dtype=torch.long)
        y = {
            'was_in_icu': torch.tensor(int(row['was_in_icu']), dtype=torch.float),
            'hosp_to_icu_1': torch.tensor(float(row['hosp_to_icu_1'])/max(1.0, delay_max), dtype=torch.float)
        }
        return x_cat, y, idx

def collate_admission(batch):
    x_cats, ys, idxs = zip(*batch)
    x_cat = torch.stack(x_cats)
    y_batch = {k: torch.stack([y[k] for y in ys]) for k in ys[0]}
    return {'x_cat': x_cat, **y_batch, 'idxs': list(idxs)}

# -------------------------
# --- Sequence encoder (GRU) to summarize all admissions per patient ---
# -------------------------
class PatientSeqEncoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=32, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=dropout if num_layers>1 else 0.0)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, seq_padded, lengths):
        lengths = torch.tensor(lengths, dtype=torch.long, device=seq_padded.device)
        packed = nn.utils.rnn.pack_padded_sequence(seq_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed)
        if self.gru.bidirectional:
            h_n = h_n.view(self.gru.num_layers, 2, seq_padded.size(0), self.gru.hidden_size)
            last = torch.cat([h_n[-1,0], h_n[-1,1]], dim=1)
        else:
            last = h_n[-1]
        return last

# -------------------------
# --- Patient GNN (Stage2) ---
# -------------------------
class PatientGNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=64, num_classes=4, dropedge_rate=0.2, jk_mode='cat'):
        super().__init__()
        self.dropedge_rate = dropedge_rate
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.jump = JumpingKnowledge(mode=jk_mode)
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim*2 + 1, 32), nn.ReLU(), nn.Linear(32,1))
        in_dim = hidden_dim * (3 if jk_mode == 'cat' else 1)
        self.heads = nn.ModuleDict({
            'mort': nn.Linear(in_dim, 1),
            'hours': nn.Linear(in_dim, 1),
            'disc_loc': nn.Linear(in_dim, num_classes)
        })

    def forward(self, data: GeoData):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # ensure dtype consistency
        x = x.float()
        if self.training and self.dropedge_rate > 0 and edge_index.numel() > 0:
            mask_e = torch.rand(edge_index.size(1), device=edge_index.device) > self.dropedge_rate
            edge_index = edge_index[:, mask_e]
            edge_weight = edge_weight[mask_e] if edge_weight is not None else None
        x_proj = self.input_proj(x)
        h1 = F.relu(self.norm1(self.conv1(x, edge_index, edge_weight)))
        h_init = h1 + x_proj
        if edge_weight is not None and edge_index.numel() > 0:
            src, dst = edge_index
            feat_cat = torch.cat([h_init[src], h_init[dst], edge_weight.unsqueeze(-1)], dim=1)
            edge_weight = torch.sigmoid(self.edge_mlp(feat_cat)).squeeze(-1)
        h2 = F.relu(self.norm2(self.conv2(h_init, edge_index))) + h_init
        h3 = F.relu(self.norm3(self.conv3(h2, edge_index))) + h2
        x_jk = self.jump([h_init, h2, h3])
        node_mask = data.node_mask.to(x.device)
        readout = x_jk[node_mask]
        mort = self.heads['mort'](readout).squeeze(-1)
        hours = self.heads['hours'](readout).squeeze(-1)
        disc = self.heads['disc_loc'](readout)
        return mort, hours, disc

# -------------------------
# --- Utilities: build patient lists and patient DF with last admission and aggregates ---
# -------------------------
def build_patient_index_lists(adm_df):
    """Return dict subject_id -> list of admission indices (sorted by year asc)."""
    subj2idxs = {}
    sorted_df = adm_df.sort_values(['subject_id', 'year']).reset_index()
    for _, row in sorted_df.iterrows():
        orig_idx = int(row['index'])
        subj = row['subject_id']
        subj2idxs.setdefault(subj, []).append(orig_idx)
    return subj2idxs

def build_patient_summary_df(adm_df, subj2idxs):
    """Construct DataFrame with one row per patient (last admission) plus aggregates."""
    rows = []
    for subj, idxs in subj2idxs.items():
        sub_adms = adm_df.loc[idxs]
        last = sub_adms.sort_values('year').iloc[-1]
        row = last.to_dict()
        row['num_adm'] = len(sub_adms)
        row['ever_in_icu'] = int(sub_adms['was_in_icu'].fillna(0).astype(int).any())
        avg_v = sub_adms[vital_cols].fillna(vitals_fill).mean().to_dict()
        for k, v in avg_v.items():
            row[f'avg_{k}'] = float(v)
        rows.append(row)
    patient_df = pd.DataFrame(rows).reset_index(drop=True)
    return patient_df

# -------------------------
# --- Build patient graph from node feature tensor and icu_probs/flags ---
# -------------------------
def build_graph_from_node_features(x_nodes, icu_probs, was_in_icu_flags, k=5, use_similarity=True, use_icutransfer=True):
    """
    x_nodes: torch.Tensor [N, D] float features for each patient node
    icu_probs: array-like length N (probabilities for last admission)
    was_in_icu_flags: array-like length N (0/1)
    returns GeoData
    """
    x = x_nodes.clone().float()
    N = x.size(0)
    edge_index_list = []
    edge_weight_list = []

    if use_similarity and N > 1:
        x_np = x.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=min(k+1, N)).fit(x_np)
        dist, idx = nbrs.kneighbors(x_np)
        for i in range(N):
            for j, d in zip(idx[i][1:], dist[i][1:]):
                edge_index_list.append([i, j])
                edge_weight_list.append(1.0 / (1.0 + d))

    if use_icutransfer:
        icu_node = N
        x = torch.cat([x, x.new_zeros((1, x.size(1)))], dim=0)
        for i, p in enumerate(icu_probs):
            edge_index_list.append([i, icu_node])
            edge_weight_list.append(float(p))

    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

    data = GeoData(x=x, edge_index=edge_index, edge_weight=edge_weight)
    # node_mask: True for patient nodes, False for dummy(if present)
    node_mask = torch.tensor([True] * N + ([False] if use_icutransfer else []), dtype=torch.bool)
    data.node_mask = node_mask
    return data

# -------------------------
# --- Main training & pipeline ---
# -------------------------
if __name__ == "__main__":
    device = torch.device('cpu')

    # --- Split admission-level df for Stage1 training/val/test ---
    train_adm_df, test_adm_df = train_test_split(df, test_size=0.1, random_state=42)
    train_adm_df, val_adm_df = train_test_split(train_adm_df, test_size=0.1, random_state=1)

    # Admission datasets and loaders
    train_adm_ds = AdmissionDataset(train_adm_df)
    val_adm_ds   = AdmissionDataset(val_adm_df)
    test_adm_ds  = AdmissionDataset(test_adm_df)

    train_loader_adm = DataLoader(train_adm_ds, batch_size=256, shuffle=True, collate_fn=collate_admission)
    val_loader_adm   = DataLoader(val_adm_ds, batch_size=128, shuffle=False, collate_fn=collate_admission)
    test_loader_adm  = DataLoader(test_adm_ds, batch_size=128, shuffle=False, collate_fn=collate_admission)

    # Instantiate Stage1 estimator
    n_cats = [int(df[c].max())+1 for c in cat_cols]
    latent_dim = 32
    estimator = ClinicalStateEstimator(n_categories=n_cats, cat_emb_dim=32, latent_dim=latent_dim).to(device)
    optim1 = torch.optim.Adam(estimator.parameters(), lr=1e-4)
    stage1_ckpt = 'estimator_admission.pt'

    # Train Stage1 if needed
    if os.path.exists(stage1_ckpt):
        estimator.load_state_dict(torch.load(stage1_ckpt, map_location=device))
        print("Loaded Stage1 from checkpoint.")
    else:
        print("Training Stage1 on admission-level data...")
        best_val = float('inf'); wait=0; patience=5
        for epoch in range(1, 31):
            estimator.train()
            tot=0.0
            for batch in train_loader_adm:
                x_cat = batch['x_cat'].to(device)
                preds = estimator(x_cat)
                loss = F.binary_cross_entropy_with_logits(preds['icu_risk'], batch['was_in_icu'].to(device)) + \
                       F.mse_loss(preds['delay'], batch['hosp_to_icu_1'].to(device))
                optim1.zero_grad(); loss.backward(); optim1.step()
                tot += loss.item()
            avg = tot / len(train_loader_adm)
            # val
            estimator.eval()
            vloss = 0.0
            with torch.no_grad():
                for batch in val_loader_adm:
                    x_cat = batch['x_cat'].to(device)
                    preds = estimator(x_cat)
                    l = F.binary_cross_entropy_with_logits(preds['icu_risk'], batch['was_in_icu'].to(device)) + \
                        F.mse_loss(preds['delay'], batch['hosp_to_icu_1'].to(device))
                    vloss += l.item()
            avgv = vloss / len(val_loader_adm)
            print(f"Epoch {epoch} TrainLoss {avg:.4f} ValLoss {avgv:.4f}")
            if avgv < best_val:
                best_val = avgv; wait = 0
                torch.save(estimator.state_dict(), stage1_ckpt)
                print("Saved best Stage1.")
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping Stage1.")
                    break
        estimator.load_state_dict(torch.load(stage1_ckpt, map_location=device))
        print("Stage1 ready.")

    estimator.eval()

    # --- Stage1 evaluation (print metrics on test_adm_df) ---
    print("Evaluating Stage1 on admission-level test set...")
    icu_logits_list = []
    icu_trues = []
    delays_pred = []
    delays_true = []
    with torch.no_grad():
        for batch in test_loader_adm:
            x_cat = batch['x_cat'].to(device)
            out = estimator(x_cat)
            icu_logits_list.extend(out['icu_risk'].cpu().numpy().tolist())
            icu_trues.extend(batch['was_in_icu'].numpy().tolist())
            delays_pred.extend(out['delay'].cpu().numpy().tolist())
            delays_true.extend(batch['hosp_to_icu_1'].numpy().tolist())
    icu_probs = torch.sigmoid(torch.tensor(icu_logits_list)).numpy()
    try:
        auc1 = roc_auc_score(icu_trues, icu_probs)
    except Exception:
        auc1 = float('nan')
    acc1 = accuracy_score(icu_trues, (icu_probs > 0.5).astype(int))
    # delays: scale back to original units
    delays_pred_orig = (np.array(delays_pred) * delay_max)
    delays_true_orig = (np.array(delays_true) * delay_max)
    mask_true_icu = np.array(icu_trues, dtype=bool)
    if mask_true_icu.sum() > 0:
        rmse_delay = np.sqrt(mean_squared_error(delays_true_orig[mask_true_icu], delays_pred_orig[mask_true_icu]))
        mae_delay = mean_absolute_error(delays_true_orig[mask_true_icu], delays_pred_orig[mask_true_icu])
    else:
        rmse_delay = float('nan'); mae_delay = float('nan')
    print(f"Stage1 Test: ICU AUC: {auc1:.4f} | Acc: {acc1:.4f} | Delay RMSE (hrs, only true ICU): {rmse_delay:.2f} | MAE: {mae_delay:.2f}")

    # --- Compute Stage1 outputs (latent, icu_risk, delay) for ALL admissions once ---
    print("Computing Stage1 latents for all admissions (this will be reused to build patient sequences)...")
    full_adm_ds = AdmissionDataset(df)  # admission dataset covering full df
    full_loader = DataLoader(full_adm_ds, batch_size=512, shuffle=False, collate_fn=collate_admission)
    N_adm = len(df)
    latents_all = np.zeros((N_adm, latent_dim), dtype=np.float32)
    icu_logits_all = np.zeros((N_adm,), dtype=np.float32)
    delay_all = np.zeros((N_adm,), dtype=np.float32)
    estimator.to(device)
    with torch.no_grad():
        for batch in full_loader:
            x_cat = batch['x_cat'].to(device)
            idxs = batch['idxs']  # list of original indices
            out = estimator(x_cat)
            lats = out['latent'].cpu().numpy()
            il = out['icu_risk'].cpu().numpy()
            dl = out['delay'].cpu().numpy()
            for i, idx in enumerate(idxs):
                latents_all[idx] = lats[i].astype(np.float32)
                icu_logits_all[idx] = float(il[i])
                delay_all[idx] = float(dl[i])

    # --- Build patient index lists and patient-level DF (last admission + aggregates) ---
    df_reset = df.reset_index(drop=True)
    subj2idxs = build_patient_index_lists(df_reset)
    patient_df = build_patient_summary_df(df_reset, subj2idxs)
    patient_adm_idxs = []
    for subj in patient_df['subject_id']:
        idxs = subj2idxs[int(subj)]
        patient_adm_idxs.append(idxs)

    # --- Prepare patient-level splits (by patient) ---
    patient_subjects = patient_df['subject_id'].values
    train_subj, test_subj = train_test_split(patient_subjects, test_size=0.1, random_state=42)
    train_subj, val_subj = train_test_split(train_subj, test_size=0.1, random_state=1)

    train_pat_df = patient_df[patient_df['subject_id'].isin(train_subj)].reset_index(drop=True)
    val_pat_df   = patient_df[patient_df['subject_id'].isin(val_subj)].reset_index(drop=True)
    test_pat_df  = patient_df[patient_df['subject_id'].isin(test_subj)].reset_index(drop=True)

    subj_to_admidxs = subj2idxs  # mapping subject -> list of adm indices

    # -------------------------
    # --- Stage2 components ---
    # -------------------------
    seq_encoder = PatientSeqEncoder(latent_dim=latent_dim, hidden_dim=32, num_layers=1, bidirectional=False).to(device)
    node_feat_dim = len(cat_cols) + latent_dim + seq_encoder.out_dim + 3 + len(vital_cols)
    gnn = PatientGNN(node_feat_dim=node_feat_dim, hidden_dim=64, num_classes=int(df['disc_loc'].max())+1).to(device)
    optim2 = torch.optim.Adam(list(gnn.parameters()) + list(seq_encoder.parameters()), lr=1e-4)

    # Helper to create node features for a batch of patients
    def make_node_features_for_patients(batch_patient_df, batch_subject_ids):
        B = len(batch_subject_ids)
        seqs = []
        lengths = []
        last_latents = []
        icu_probs_last = []
        was_in_icu_flags = []
        labels_mort = []
        labels_hours = []
        labels_disc = []
        cats_last = []
        avg_vitals_list = []
        num_adm_list = []
        ever_in_icu_list = []
        for i, subj in enumerate(batch_subject_ids):
            subj = int(subj)
            adm_idxs = subj_to_admidxs[subj]
            seq_latents = latents_all[adm_idxs]  # numpy [T, latent_dim]
            seqs.append(torch.tensor(seq_latents, dtype=torch.float))
            lengths.append(seq_latents.shape[0])
            last_idx = adm_idxs[-1]
            last_latents.append(torch.tensor(latents_all[last_idx], dtype=torch.float))
            icu_logits_last = icu_logits_all[last_idx]
            icu_probs_last.append(float(torch.sigmoid(torch.tensor(icu_logits_last)).item()))
            was_in_icu_flags.append(int(df_reset.loc[last_idx,'was_in_icu']) if 'was_in_icu' in df_reset.columns else 0)
            labels_mort.append(float(df_reset.loc[last_idx,'in_hosp_mortality']))
            hrs = float(df_reset.loc[last_idx]['icu_duration_1']) if not np.isnan(df_reset.loc[last_idx]['icu_duration_1']) else 0.0
            labels_hours.append(hrs / max(1.0, dur_max))
            labels_disc.append(int(df_reset.loc[last_idx]['disc_loc']))
            cats_last.append([float(df_reset.loc[last_idx, c]) for c in cat_cols])
            row = batch_patient_df.iloc[i]
            avg_v = [float(row[f'avg_{v}']) for v in vital_cols]
            avg_vitals_list.append(avg_v)
            num_adm_list.append(float(row['num_adm']))
            ever_in_icu_list.append(float(row['ever_in_icu']))
        max_len = max(lengths)
        seq_padded = torch.zeros((B, max_len, latent_dim), dtype=torch.float)
        for i, s in enumerate(seqs):
            seq_padded[i, :s.size(0), :] = s
        seq_padded = seq_padded.to(device)
        seq_encoder.to(device)
        seq_emb = seq_encoder(seq_padded, lengths).to(device)
        cats_tensor = torch.tensor(cats_last, dtype=torch.float, device=device)
        last_lat_tensor = torch.stack(last_latents).to(device)
        num_adm_t = torch.tensor(num_adm_list, dtype=torch.float, device=device).unsqueeze(1)
        ever_in_icu_t = torch.tensor(ever_in_icu_list, dtype=torch.float, device=device).unsqueeze(1)
        last_was_icu_t = torch.tensor(was_in_icu_flags, dtype=torch.float, device=device).unsqueeze(1)
        avg_vitals = torch.tensor(avg_vitals_list, dtype=torch.float, device=device)
        avg_vitals_norm = (avg_vitals - torch.tensor(vitals_mean.values, device=device).unsqueeze(0)) / torch.tensor(vitals_std.values, device=device).unsqueeze(0)
        x_nodes = torch.cat([cats_tensor, last_lat_tensor, seq_emb, num_adm_t, ever_in_icu_t, last_was_icu_t, avg_vitals_norm], dim=1)
        labels_mort = torch.tensor(labels_mort, dtype=torch.float, device=device)
        labels_hours = torch.tensor(labels_hours, dtype=torch.float, device=device)
        labels_disc = torch.tensor(labels_disc, dtype=torch.long, device=device)
        return x_nodes, icu_probs_last, was_in_icu_flags, labels_mort, labels_hours, labels_disc

    # --- Stage2 training prep ---
    train_idx = list(train_pat_df.index)
    val_idx = list(val_pat_df.index)
    test_idx = list(test_pat_df.index)

    batch_size_pat = 32
    stage2_ckpt = 'gnn_patient_seq.pt'
    print("Start Stage2 training (patient-level graph with sequence info)...")
    if os.path.exists(stage2_ckpt):
        ckpt = torch.load(stage2_ckpt, map_location=device)
        gnn.load_state_dict(ckpt['gnn'])
        seq_encoder.load_state_dict(ckpt['seq_enc'])
        print("Loaded Stage2 checkpoint.")
    else:
        best_val = float('inf'); wait=0; patience=5
        epochs2 = 150
        for epoch in range(1, epochs2+1):
            gnn.train(); seq_encoder.train()
            total_loss = 0.0
            random.shuffle(train_idx)
            n_batches = max(1, (len(train_idx) + batch_size_pat - 1) // batch_size_pat)
            for i in range(0, len(train_idx), batch_size_pat):
                batch_ids = train_idx[i:i+batch_size_pat]
                batch_patient_df = train_pat_df.iloc[batch_ids].reset_index(drop=True)
                batch_subjects = batch_patient_df['subject_id'].values.tolist()
                x_nodes, icu_probs_last, was_in_icu_flags, y_mort, y_hours, y_disc = make_node_features_for_patients(batch_patient_df, batch_subjects)
                data = build_graph_from_node_features(x_nodes, icu_probs_last, was_in_icu_flags, k=15, use_similarity=True, use_icutransfer=True)
                data = data.to(device)
                data.y_mort = y_mort
                data.y_hours = y_hours
                data.y_disc = y_disc
                data.is_icu_node = torch.tensor(was_in_icu_flags, dtype=torch.bool, device=device)
                # node_mask already set inside build_graph; ensure on device
                data.node_mask = data.node_mask.to(device)
                pred_mort, pred_hours, pred_disc = gnn(data)
                mask = data.is_icu_node
                l_mort = F.binary_cross_entropy_with_logits(pred_mort, data.y_mort)
                l_disc = F.cross_entropy(pred_disc, data.y_disc)
                l_dur = F.mse_loss(pred_hours[mask], data.y_hours[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=device)
                loss = l_mort + l_disc + l_dur
                optim2.zero_grad()
                loss.backward()
                optim2.step()
                total_loss += loss.item()
            avg_train = total_loss / float(n_batches)
            # validation
            gnn.eval(); seq_encoder.eval()
            val_loss = 0.0
            n_val_batches = max(1, (len(val_idx) + batch_size_pat - 1) // batch_size_pat)
            with torch.no_grad():
                for i in range(0, len(val_idx), batch_size_pat):
                    batch_ids = val_idx[i:i+batch_size_pat]
                    batch_patient_df = val_pat_df.iloc[batch_ids].reset_index(drop=True)
                    batch_subjects = batch_patient_df['subject_id'].values.tolist()
                    x_nodes, icu_probs_last, was_in_icu_flags, y_mort, y_hours, y_disc = make_node_features_for_patients(batch_patient_df, batch_subjects)
                    data = build_graph_from_node_features(x_nodes, icu_probs_last, was_in_icu_flags, k=5, use_similarity=True, use_icutransfer=True)
                    data = data.to(device)
                    data.y_mort = y_mort
                    data.y_hours = y_hours
                    data.y_disc = y_disc
                    data.is_icu_node = torch.tensor(was_in_icu_flags, dtype=torch.bool, device=device)
                    data.node_mask = data.node_mask.to(device)
                    pred_mort, pred_hours, pred_disc = gnn(data)
                    mask = data.is_icu_node
                    l_mort = F.binary_cross_entropy_with_logits(pred_mort, data.y_mort)
                    l_disc = F.cross_entropy(pred_disc, data.y_disc)
                    l_dur = F.mse_loss(pred_hours[mask], data.y_hours[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=device)
                    val_loss += (l_mort + l_disc + l_dur).item()
            avg_val = val_loss / float(n_val_batches)
            print(f"[Epoch {epoch}] TrainLoss {avg_train:.4f} ValLoss {avg_val:.4f}")
            if avg_val < best_val:
                best_val = avg_val; wait = 0
                torch.save({'gnn': gnn.state_dict(), 'seq_enc': seq_encoder.state_dict()}, stage2_ckpt)
                print("Saved best Stage2 checkpoint.")
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping Stage2.")
                    break

    # --- Final evaluation on test_pat_df ---
    print("Evaluating Stage2 on test patients...")
    ckpt = torch.load(stage2_ckpt, map_location=device)
    gnn.load_state_dict(ckpt['gnn']); seq_encoder.load_state_dict(ckpt['seq_enc'])
    gnn.eval(); seq_encoder.eval()

    all_mort_probs = []
    all_mort_trues = []
    all_hours_preds = []
    all_hours_trues = []
    all_disc_preds = []
    all_disc_trues = []

    with torch.no_grad():
        for i in range(0, len(test_pat_df), batch_size_pat):
            batch_patient_df = test_pat_df.iloc[i:i+batch_size_pat].reset_index(drop=True)
            batch_subjects = batch_patient_df['subject_id'].values.tolist()

            x_nodes, icu_probs_last, was_in_icu_flags, y_mort, y_hours, y_disc = \
                make_node_features_for_patients(batch_patient_df, batch_subjects)

            data = build_graph_from_node_features(
                x_nodes, icu_probs_last, was_in_icu_flags,
                k=5, use_similarity=True, use_icutransfer=True
            )
            data = data.to(device)
            data.y_mort = y_mort
            data.y_hours = y_hours
            data.y_disc = y_disc
            data.is_icu_node = torch.tensor(was_in_icu_flags, dtype=torch.bool, device=device)
            data.node_mask = data.node_mask.to(device)

            mort_logits, hours_p, disc_p = gnn(data)

            probs = torch.sigmoid(mort_logits).cpu().numpy()
            all_mort_probs.extend(probs.tolist())
            all_mort_trues.extend(data.y_mort.cpu().numpy().tolist())

            mask = data.is_icu_node
            if mask.sum() > 0:
                all_hours_preds.extend((hours_p[mask].cpu().numpy() * dur_max).tolist())
                all_hours_trues.extend((data.y_hours[mask].cpu().numpy() * dur_max).tolist())

            all_disc_preds.extend(disc_p.argmax(dim=1).cpu().numpy().tolist())
            all_disc_trues.extend(data.y_disc.cpu().numpy().tolist())

    # --- Metriche classiche ---
    try:
        auc_m = roc_auc_score(all_mort_trues, all_mort_probs)
    except Exception:
        auc_m = float('nan')
    acc_m = accuracy_score(all_mort_trues, (np.array(all_mort_probs) > 0.5).astype(int))
    print(f"Mortality AUC: {auc_m:.4f}, Acc: {acc_m:.4f}")

    if len(all_hours_trues) > 0:
        rmse_icu = np.sqrt(mean_squared_error(all_hours_trues, all_hours_preds))
        mae_icu = mean_absolute_error(all_hours_trues, all_hours_preds)
    else:
        rmse_icu = float('nan'); mae_icu = float('nan')
    print(f"ICU duration RMSE (hours): {rmse_icu:.2f}, MAE: {mae_icu:.2f}")

    if len(all_disc_trues) > 0:
        disc_acc = accuracy_score(all_disc_trues, all_disc_preds)
    else:
        disc_acc = float('nan')
    print(f"Discharge location accuracy: {disc_acc:.4f}")

    # -----------------------------
    # Confusion matrices (Stage 2)
    # -----------------------------
    # Mortality
    mort_true_arr = np.array(all_mort_trues, dtype=int)
    mort_pred_arr = (np.array(all_mort_probs) > 0.5).astype(int)
    cm_mort = confusion_matrix(mort_true_arr, mort_pred_arr)

    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    im = ax.imshow(cm_mort, interpolation='nearest', cmap='Blues')
    ax.set_title("Confusion Matrix – In-hospital Mortality")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['0', '1'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # annotazioni
    for i in range(cm_mort.shape[0]):
        for j in range(cm_mort.shape[1]):
            ax.text(
                j, i, str(cm_mort[i, j]),
                ha="center", va="center",
                color="white" if cm_mort[i, j] > cm_mort.max() / 2.0 else "black"
            )
    plt.tight_layout()
    plt.show()

    # Discharge location
    disc_true_arr = np.array(all_disc_trues, dtype=int)
    disc_pred_arr = np.array(all_disc_preds, dtype=int)
    cm_disc = confusion_matrix(disc_true_arr, disc_pred_arr)

    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(cm_disc, interpolation='nearest', cmap='Greens')
    ax.set_title("Confusion Matrix – Discharge Location")
    plt.colorbar(im, ax=ax)

    num_classes = cm_disc.shape[0]
    ax.set_xticks(list(range(num_classes)))
    ax.set_yticks(list(range(num_classes)))
    ax.set_xticklabels([str(i) for i in range(num_classes)])
    ax.set_yticklabels([str(i) for i in range(num_classes)])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    for i in range(cm_disc.shape[0]):
        for j in range(cm_disc.shape[1]):
            ax.text(
                j, i, str(cm_disc[i, j]),
                ha="center", va="center",
                color="white" if cm_disc[i, j] > cm_disc.max() / 2.0 else "black"
            )
    plt.tight_layout()
    plt.show()

    # ---------------------------------
    # Explainability (grad-based simple)
    # ---------------------------------
    print("\nRunning simple gradient-based explainability on one test patient...")

    # prendiamo il primo batch di test
    if len(test_pat_df) > 0:
        batch_patient_df = test_pat_df.iloc[0:batch_size_pat].reset_index(drop=True)
        batch_subjects = batch_patient_df['subject_id'].values.tolist()

        x_nodes, icu_probs_last, was_in_icu_flags, y_mort, y_hours, y_disc = \
            make_node_features_for_patients(batch_patient_df, batch_subjects)

        data_exp = build_graph_from_node_features(
            x_nodes, icu_probs_last, was_in_icu_flags,
            k=5, use_similarity=True, use_icutransfer=True
        )
        data_exp = data_exp.to(device)
        data_exp.node_mask = data_exp.node_mask.to(device)
        data_exp.is_icu_node = torch.tensor(was_in_icu_flags, dtype=torch.bool, device=device)

        # abilitiamo il gradiente sulle feature dei nodi
        data_exp.x = data_exp.x.clone().detach().requires_grad_(True)

        gnn.zero_grad()
        pred_mort_exp, pred_hours_exp, pred_disc_exp = gnn(data_exp)

        # mappiamo indice globale nodo -> indice in readout
        patient_nodes = torch.where(data_exp.node_mask)[0]
        if len(patient_nodes) > 0:
            # scegliamo il primo paziente del batch
            target_global_idx = patient_nodes[0]
            target_readout_idx = 0  # perché readout è nell'ordine dei True in node_mask

            target_logit = pred_mort_exp[target_readout_idx]
            target_logit.backward()

            # gradiente rispetto alle feature del nodo target
            node_grad = data_exp.x.grad[target_global_idx].detach().cpu().numpy()
            feat_importance = np.abs(node_grad)

            # helper per spiegare gli indici delle feature
            def describe_feature_index(i):
                offset = 0
                # categorie
                if i < len(cat_cols):
                    return f"cat:{cat_cols[i]}"
                offset += len(cat_cols)
                # latent stage1
                if i < offset + latent_dim:
                    return f"stage1_latent_dim_{i - offset}"
                offset += latent_dim
                # seq encoder
                if i < offset + seq_encoder.out_dim:
                    return f"seq_emb_dim_{i - offset}"
                offset += seq_encoder.out_dim
                # num_adm, ever_in_icu, last_was_in_icu
                if i == offset:
                    return "num_adm"
                if i == offset + 1:
                    return "ever_in_icu"
                if i == offset + 2:
                    return "last_was_in_icu"
                offset += 3
                # avg vitals
                if i < offset + len(vital_cols):
                    return f"avg_{vital_cols[i - offset]}"
                return f"unknown_{i}"

            top_k = 15
            top_idx = np.argsort(-feat_importance)[:top_k]

            print("\nTop feature importances for selected patient (grad |d logit / dx|):")
            for i in top_idx:
                print(f"  idx {i:3d} ({describe_feature_index(i)}): {feat_importance[i]:.6f}")

            # importanza degli archi collegati al nodo target (in base al peso appreso)
            edge_index = data_exp.edge_index
            edge_weight = data_exp.edge_weight.detach().cpu().numpy() if data_exp.edge_weight is not None else None

            if edge_weight is not None and edge_index.size(1) > 0:
                num_pat_nodes = int(data_exp.node_mask.sum().item())
                icu_dummy_idx = num_pat_nodes if data_exp.x.size(0) > num_pat_nodes else None

                mask_edges = (edge_index[0] == target_global_idx) | (edge_index[1] == target_global_idx)
                idx_edges = torch.where(mask_edges)[0].cpu().numpy()

                if len(idx_edges) > 0:
                    edge_w_sel = edge_weight[idx_edges]
                    order = np.argsort(-edge_w_sel)
                    print("\nTop incident edges for selected patient (by learned edge weight):")
                    for k in range(min(10, len(order))):
                        e_id = idx_edges[order[k]]
                        src = int(edge_index[0, e_id].item())
                        dst = int(edge_index[1, e_id].item())
                        w = edge_weight[e_id]
                        def node_desc(n):
                            if icu_dummy_idx is not None and n == icu_dummy_idx:
                                return f"node {n} (ICU dummy hub)"
                            elif n < num_pat_nodes:
                                return f"node {n} (patient in batch idx {n})"
                            else:
                                return f"node {n}"
                        print(f"  edge {k+1}: {node_desc(src)} -> {node_desc(dst)}, weight={w:.4f}")
                else:
                    print("\nNo edges incident to selected patient node.")
        else:
            print("No patient nodes found in explainability batch.")

    print("Done.")
