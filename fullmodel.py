# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:29:35 2025

@author: admin
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, GroupKFold
from collections import Counter
from itertools import combinations
from modeltransfer import emsc

# --- 1) EMSC Preprocessing ---
def preemsc2(X_train_df, y_train, X_test_df, df, train_index):
    X_train = X_train_df.values.astype(np.float64)
    X_test = X_test_df.values.astype(np.float64)
    ref = df.iloc[:, 2:].values.astype(np.float64).mean(axis=0)

    X_train_emsc = np.zeros_like(X_train, dtype=np.float64)
    for label in np.unique(y_train):
        mask = (y_train == label)
        Xi = X_train[mask]
        pats = df['patient'].iloc[train_index].values[mask]
        uniq = np.unique(pats)
        mean_spectra = np.vstack([Xi[pats==p].mean(axis=0) for p in uniq])
        pca_emsc = PCA(n_components=3)
        pca_emsc.fit(mean_spectra)
        interfere = pca_emsc.components_[0].reshape(1, -1).astype(np.float64)

        res = emsc(Xi, degree=4, reference=ref, interferent=interfere,
                  constituent=None, wn=None, interf_pca=0, contit_pca=0)
        X_train_emsc[mask] = res['corrected']

    itrain = X_train_emsc.mean(axis=0, keepdims=True)
    itest = X_test.mean(axis=0, keepdims=True)
    interferent = np.vstack([itrain, itest]).astype(np.float64)
    res_test = emsc(X_test, degree=0, reference=ref, interferent=interferent,
                    constituent=None, wn=None, interf_pca=1, contit_pca=0)
    X_test_emsc = res_test['corrected']

    Xtr_df = pd.DataFrame(X_train_emsc, index=X_train_df.index, columns=X_train_df.columns)
    Xte_df = pd.DataFrame(X_test_emsc, index=X_test_df.index, columns=X_test_df.columns)
    return Xtr_df, Xte_df, interferent

# --- 2) Raw feature fusion ---
def raw_feature_fusion(dfs):
    base = next(iter(dfs.values()))[['patient', 'group']].copy()
    out = base.copy()
    for name, df in dfs.items():
        feats = df.drop(columns=['patient', 'group']).copy()
        feats.columns = [f"{name}_{c}" for c in feats.columns]
        out = pd.concat([out, feats.reset_index(drop=True)], axis=1)
    return out

# --- 3) PCA feature fusion ---
def pca_feature_fusion(dfs, pre_pca_components):
    base = next(iter(dfs.values()))[['patient', 'group']].copy()
    out = base.copy()
    for name, df in dfs.items():
        X = df.drop(columns=['patient', 'group']).values.astype(float)
        pca = PCA(n_components=pre_pca_components)
        Xp = pca.fit_transform(X)
        cols = [f"{name}_PC{i+1}" for i in range(pre_pca_components)]
        pcd = pd.DataFrame(Xp, columns=cols, index=df.index)
        out = pd.concat([out, pcd.reset_index(drop=True)], axis=1)
    return out

# --- 4) CV splitter ---
def get_splitter(cv_type, n_splits, X, y, groups=None):
    if cv_type == 'kfold':
        return KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y)
    else:
        return GroupKFold(n_splits=n_splits).split(X, y, groups)

# --- 5) Fold evaluation: PCA -> LDA -> patient vote ---
def _fold_eval_with_pca(df, train_idx, test_idx, n_components, preprocessing):
    Xdf = df.drop(columns=['patient', 'group'])
    y = df['group'].values
    pats = df['patient'].values
    Xtr_df, Xte_df = Xdf.iloc[train_idx], Xdf.iloc[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    pats_te = pats[test_idx]

    if preprocessing == 'emsc':
        Xtr, Xte, _ = preemsc2(Xtr_df, ytr, Xte_df, df, train_idx)
    else:
        Xtr, Xte = Xtr_df.values, Xte_df.values

    pca = PCA(n_components=n_components)
    Xtr_p = pca.fit_transform(Xtr)
    Xte_p = pca.transform(Xte)
    y_pred = LDA().fit(Xtr_p, ytr).predict(Xte_p)

    vote_map = {}
    for pid, true, pred in zip(pats_te, yte, y_pred):
        vote_map.setdefault(pid, {'true': true, 'preds': []})['preds'].append(pred)
    y_true_f = [v['true'] for v in vote_map.values()]
    y_pred_f = [Counter(v['preds']).most_common(1)[0][0] for v in vote_map.values()]

    return balanced_accuracy_score(y_true_f, y_pred_f), y_true_f, y_pred_f

# --- 6) Raw-level PCA -> LDA evaluation ---
def evaluate_pca_lda_folds(df, n_components, cv_type, n_splits, preprocessing):
    X = df.drop(columns=['patient', 'group']).values
    y = df['group'].values
    groups = df['patient'].values if cv_type == 'groupkfold' else None
    split_gen = get_splitter(cv_type, n_splits, X, y, groups)

    fold_accs, all_true, all_pred = [], [], []
    for tr, te in split_gen:
        acc, yt, yp = _fold_eval_with_pca(df, tr, te, n_components, preprocessing)
        fold_accs.append(acc)
        all_true.extend(yt)
        all_pred.extend(yp)
    return fold_accs, all_true, all_pred

# --- 7) PCA-level direct LDA evaluation ---
def evaluate_lda_folds(df, cv_type, n_splits, preprocessing):
    X = df.drop(columns=['patient', 'group']).values
    y = df['group'].values
    groups = df['patient'].values if cv_type == 'groupkfold' else None
    split_gen = get_splitter(cv_type, n_splits, X, y, groups)

    fold_accs, all_true, all_pred = [], [], []
    for tr, te in split_gen:
        Xtr, Xte = (preemsc2(df.drop(columns=['patient', 'group']).iloc[tr], y[tr],
                             df.drop(columns=['patient', 'group']).iloc[te], df, tr)[:2]
                    if preprocessing == 'emsc'
                    else (X[tr], X[te]))
        ytr, yte = y[tr], y[te]
        y_pred = LDA().fit(Xtr, ytr).predict(Xte)

        pats_te = df['patient'].values[te]
        vote_map = {}
        for pid, true, pred in zip(pats_te, yte, y_pred):
            vote_map.setdefault(pid, {'true': true, 'preds': []})['preds'].append(pred)
        y_true_f = [v['true'] for v in vote_map.values()]
        y_pred_f = [Counter(v['preds']).most_common(1)[0][0] for v in vote_map.values()]

        fold_accs.append(balanced_accuracy_score(y_true_f, y_pred_f))
        all_true.extend(y_true_f)
        all_pred.extend(y_pred_f)
    return fold_accs, all_true, all_pred

# --- 8) Model-level fusion evaluation ---
def model_level_fusion_folds(dfs, n_components, cv_type, n_splits, preprocessing):
    base = next(iter(dfs.values()))
    X0 = base.drop(columns=['patient', 'group']).values
    y0 = base['group'].values
    groups0 = base['patient'].values
    split_gen = get_splitter(cv_type, n_splits, X0, y0, groups0)

    fold_accs, excluded_idxs, all_true, all_pred = [], [], [], []
    for tr, te in split_gen:
        preds = {}
        for name, df in dfs.items():
            df_tr = df.drop(columns=['patient', 'group']).iloc[tr]
            df_te = df.drop(columns=['patient', 'group']).iloc[te]
            ytr = df['group'].values[tr]

            if preprocessing == 'emsc':
                Xtr, Xte, _ = preemsc2(df_tr, ytr, df_te, df, tr)
            else:
                Xtr, Xte = df_tr.values, df_te.values

            pca = PCA(n_components=n_components)
            Xtr_p = pca.fit_transform(Xtr)
            Xte_p = pca.transform(Xte)
            preds[name] = LDA().fit(Xtr_p, ytr).predict(Xte_p)

        pred_df = pd.DataFrame(preds, index=te)
        mask = pred_df.eq(pred_df.iloc[:, 0], axis=0).all(axis=1)
        keep = pred_df.index[mask]
        excl = pred_df.index[~mask].tolist()
        excluded_idxs.append(excl)

        vote_map = {}
        for idx in keep:
            pid = base.loc[idx, 'patient']
            true = base.loc[idx, 'group']
            p_ = pred_df.loc[idx].iloc[0]
            vote_map.setdefault(pid, {'true': true, 'preds': []})['preds'].append(p_)
        y_true_f = [v['true'] for v in vote_map.values()]
        y_pred_f = [Counter(v['preds']).most_common(1)[0][0] for v in vote_map.values()]

        fold_accs.append(balanced_accuracy_score(y_true_f, y_pred_f))
        all_true.extend(y_true_f)
        all_pred.extend(y_pred_f)

    return fold_accs, excluded_idxs, all_true, all_pred

# --- 9) Main fusion experiment runner ---
def run_fusion_experiments(dfs, fusion_modes, cv_type,
                           pre_pca_components_list, n_components_list,
                           n_splits=5, preprocessing=None, combination_sizes=None):
    keys = list(dfs)
    summary, folds = [], []

    if combination_sizes is None:
        combination_sizes = list(range(1, len(keys) + 1))

    for r in combination_sizes:
        if r < 1 or r > len(keys):
            continue
        for combo in combinations(keys, r):
            name, sub = "+".join(combo), {k: dfs[k] for k in combo}

            if 'raw' in fusion_modes:
                best = {'mean': -np.inf, 'std': 0, 'nc': None}
                fused = raw_feature_fusion(sub)
                for nc in n_components_list:
                    accs, _, _ = evaluate_pca_lda_folds(fused, nc, cv_type, n_splits, preprocessing)
                    folds += [{'combination': name, 'keys': combo, 'fusion': 'raw',
                               'n_components': nc, 'accuracy': a} for a in accs]
                    m, s = np.nanmean(accs), np.nanstd(accs)
                    if m > best['mean']:
                        best.update({'mean': m, 'std': s, 'nc': nc})
                summary.append({'combination': name, 'keys': combo, 'fusion': 'raw',
                                'best_n_components': best['nc'],
                                'mean_accuracy': best['mean'], 'std_accuracy': best['std']})

            if 'pca' in fusion_modes:
                best = {'mean': -np.inf, 'std': 0, 'pc': None}
                for pc in pre_pca_components_list:
                    fused_p = pca_feature_fusion(sub, pc)
                    accs, _, _ = evaluate_lda_folds(fused_p, cv_type, n_splits, preprocessing)
                    folds += [{'combination': name, 'keys': combo, 'fusion': 'pca',
                               'pre_pca': pc, 'accuracy': a} for a in accs]
                    m, s = np.nanmean(accs), np.nanstd(accs)
                    if m > best['mean']:
                        best.update({'mean': m, 'std': s, 'pc': pc})
                summary.append({'combination': name, 'keys': combo, 'fusion': 'pca',
                                'best_pre_pca': best['pc'],
                                'mean_accuracy': best['mean'], 'std_accuracy': best['std']})

            if 'model' in fusion_modes and r >= 1:
                best = {'mean': -np.inf, 'std': 0, 'nc': None}
                for nc in n_components_list:
                    accs, excl, _, _ = model_level_fusion_folds(sub, nc, cv_type, n_splits, preprocessing)
                    folds += [{'combination': name, 'keys': combo, 'fusion': 'model',
                               'n_components': nc, 'accuracy': a} for a in accs]
                    m, s = np.nanmean(accs), np.nanstd(accs)
                    if m > best['mean']:
                        best.update({'mean': m, 'std': s, 'nc': nc})
                summary.append({'combination': name, 'keys': combo, 'fusion': 'model',
                                'best_n_components': best['nc'],
                                'mean_accuracy': best['mean'], 'std_accuracy': best['std']})

    return pd.DataFrame(summary), pd.DataFrame(folds)
