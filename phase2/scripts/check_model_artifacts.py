"""Independently trace the local CSV, scaler and trained weights to saved scores.

Run after both phases: python phase2/scripts/check_model_artifacts.py
This needs the excluded dataset, arrays and weights. It does not retrain models.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
P1, P2 = ROOT / 'phase1', ROOT / 'phase2'
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
raw_path = P1 / 'data/raw/creditcard.csv'
required = [raw_path, P1/'data/processed/features.npz', P1/'models/gmm_models.joblib',
            P2/'models/features.npz', P2/'models/scaler.joblib']
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
assert not missing, f'Run both phases first. Missing local files: {missing}'

raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
assert raw_hash == json.loads((P2/'results/data_audit.json').read_text())['raw_sha256']
raw = pd.read_csv(raw_path)
clean = raw.sort_values('Time', kind='stable').drop_duplicates().reset_index(drop=True)
n = len(clean)
parts = {'train': clean.iloc[:int(.6*n)],
         'validation': clean.iloc[int(.6*n):int(.8*n)],
         'test': clean.iloc[int(.8*n):]}
assert parts['train'].Time.max() < parts['validation'].Time.min()
assert parts['validation'].Time.max() < parts['test'].Time.min()
features = json.loads((P2/'results/features.json').read_text())
assert features == [f'V{i}' for i in range(1, 29)] + ['LogAmount']

def inputs(frame):
    values = frame[[f'V{i}' for i in range(1, 29)]].copy()
    values['LogAmount'] = np.log1p(frame.Amount)
    return values

normal_train = parts['train'].loc[parts['train'].Class == 0]
assert len(normal_train) == 169893
scaler = StandardScaler().fit(inputs(normal_train))
saved_scaler = joblib.load(P2/'models/scaler.joblib')
assert saved_scaler.n_samples_seen_ == len(normal_train)
assert np.allclose(scaler.mean_, saved_scaler.mean_, rtol=1e-12, atol=1e-12)
assert np.allclose(scaler.scale_, saved_scaler.scale_, rtol=1e-12, atol=1e-12)
arrays = np.load(P2/'models/features.npz')
baseline_arrays = np.load(P1/'data/processed/features.npz')
for split, frame in [('train', normal_train), ('validation', parts['validation']), ('test', parts['test'])]:
    key = f'X_{split}'
    assert np.allclose(scaler.transform(inputs(frame)), arrays[key], rtol=1e-10, atol=1e-10)
    assert np.allclose(baseline_arrays[key], arrays[key], rtol=1e-10, atol=1e-10)
for split in ['validation', 'test']:
    assert np.array_equal(parts[split].Class.to_numpy(), arrays[f'y_{split}'])

saved = pd.read_csv(P2/'results/test_scores.csv')
assert np.array_equal(saved.Class, arrays['y_test'])
gmm = joblib.load(P1/'models/gmm_models.joblib')['selected_gmm']
grid = pd.read_csv(P1/'results/gmm_selection.csv')
assert gmm.n_components == int(grid.loc[grid.BIC.idxmin(), 'K']) == 8
assert gmm.converged_
val = {'GMM': -gmm.score_samples(arrays['X_validation'])}
test = {'GMM': -gmm.score_samples(arrays['X_test'])}
pca = PCA(n_components=8, svd_solver='full').fit(arrays['X_train'])
for target, key in [(val, 'X_validation'), (test, 'X_test')]:
    x = arrays[key]
    target['PCA8'] = ((pca.inverse_transform(pca.transform(x))-x)**2).mean(axis=1)

def errors(state, values):
    """Use saved tensors directly, independently of the notebook's model class."""
    result = []
    with torch.no_grad():
        for clean_batch in torch.from_numpy(values.astype('float32')).split(4096):
            z = clean_batch
            for layer in ['encoder.0', 'encoder.2', 'decoder.0', 'decoder.2']:
                z = F.linear(z, state[layer+'.weight'], state[layer+'.bias'])
                if layer != 'decoder.2':
                    z = F.relu(z)
            result.append(((z-clean_batch)**2).mean(dim=1).numpy())
    return np.concatenate(result)

settings = json.loads((P2/'results/training_settings.json').read_text())
normal = arrays['y_validation'] == 0
for run in settings['runs']:
    name = run['name']
    state = torch.load(P2/f'models/{name}.pt', weights_only=True, map_location='cpu')
    assert sum(value.numel() for value in state.values()) == run['parameters']
    val[name], test[name] = errors(state, arrays['X_validation']), errors(state, arrays['X_test'])
    assert np.isclose(float(val[name][normal].mean()), run['best_normal_validation_mse'], rtol=1e-6)
    history = pd.read_csv(P2/f'results/{name}_history.csv')
    best, epoch, stale = float('inf'), None, 0
    for row in history.itertuples():
        if row.normal_validation_mse < best - settings['min_delta']:
            best, epoch, stale = row.normal_validation_mse, row.epoch, 0
        else:
            stale += 1
        assert stale < settings['patience'] or row.epoch == len(history)
    assert epoch == run['best_epoch'] and len(history) == run['epochs_run']
    assert len(history) == settings['max_epochs'] or stale == settings['patience']

policy = json.loads((P2/'results/frozen_policy.json').read_text())
assert policy['hybrid_weights'] == [.5, .5]
assert not policy['model_selection_uses_fraud_labels']
for target in [val, test]:
    branches = []
    for model, branch, transform in [('GMM', 'GMM', np.arcsinh), ('denoising_ae', 'AE', np.log1p)]:
        ref = transform(val[model][normal])
        q25, median, q75 = np.percentile(ref, [25, 50, 75])
        scale = max(q75-q25, 1e-8)
        assert np.isclose(median, policy['normalization'][branch]['median'])
        assert np.isclose(scale, policy['normalization'][branch]['iqr'])
        branches.append((transform(target[model])-median)/scale)
    target['Hybrid'] = .5*branches[0] + .5*branches[1]
for name, values in test.items():
    assert np.allclose(values, saved[name], rtol=1e-6, atol=1e-7), name
    for budget, cutoff in policy['thresholds'][name].items():
        expected = np.quantile(val[name][normal], 1-float(budget), method='higher')
        assert np.isclose(expected, cutoff, rtol=1e-7, atol=1e-7), (name, budget)

draft_candidates = []
draft_path = P1/'models/draft_gmm_candidates.joblib'
if draft_path.exists():
    candidates = joblib.load(draft_path)
    draft = pd.read_csv(P1/'output/drafts/gmm_k_sweep/comparison.csv').set_index('K')
    assert sorted(candidates) == list(draft.index) == [4, 6, 8, 12, 16, 24, 32]
    for k, candidate in candidates.items():
        assert candidate.converged_
        assert np.isclose(candidate.bic(arrays['X_train']), draft.loc[k, 'BIC'])
        v = -candidate.score_samples(arrays['X_validation'])
        cutoff = np.quantile(v[normal], .995, method='higher')
        assert np.isclose(cutoff, draft.loc[k, 'Threshold'])
        for period, labels, values in [
            ('Validation', arrays['y_validation'], v),
            ('Test', arrays['y_test'], -candidate.score_samples(arrays['X_test']))]:
            p, r, _ = precision_recall_curve(labels, values)
            assert np.isclose(auc(r, p), draft.loc[k, period+' PR AUC'])
            tn, fp, fn, tp = confusion_matrix(labels, values > cutoff, labels=[0, 1]).ravel()
            assert tp == draft.loc[k, period+' fraud found']
            assert fp == draft.loc[k, period+' false alerts']
            order = np.argsort(-values, kind='stable')
            assert labels[order[:100]].sum() == draft.loc[k, period+' top 100 frauds']
        draft_candidates.append(int(k))

receipt = {'verified_at_ist': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
           'raw_sha256': raw_hash, 'strict_chronology': True,
           'legitimate_training_rows': len(normal_train), 'scaler_refitted_from_raw': True,
           'shared_arrays_checked': True, 'ae_checkpoints_reloaded': 4,
           'early_stopping_histories_replayed': True, 'models_rescored': list(test),
           'normal_validation_normalization_recomputed': True,
           'all_validation_cutoffs_recomputed': True, 'saved_test_scores_reproduced': True,
           'exploratory_gmm_candidates_rescored': draft_candidates,
           'test_scores_sha256': hashlib.sha256((P2/'results/test_scores.csv').read_bytes()).hexdigest(),
           'checkpoint_sha256': {run['name']: hashlib.sha256(
               (P2/f"models/{run['name']}.pt").read_bytes()).hexdigest()
               for run in settings['runs']}}
(P2/'results/model_artifact_verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
print('PASS: raw data, chronology, normal-only scaler, checkpoints, seven score columns and cutoffs.')
