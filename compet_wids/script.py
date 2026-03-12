import numpy as np
import pandas as pd
import warnings
import os
import time as timer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss
from scipy.optimize import minimize
import lightgbm as lgb
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, ExtraSurvivalTrees, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.util import Surv

warnings.filterwarnings('ignore')
np.random.seed(42)
HORIZONS_PRED = np.array([12, 24, 48, 72], dtype=float)

def locate_datasets():
    train_path, test_path = 'train.csv', 'test.csv'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_roots = [
        '/kaggle/input', '../input',
        os.path.join(script_dir, 'datas'),
        script_dir,
        'datas', '.',
    ]
    for search_root in search_roots:
        if os.path.exists(search_root):
            for root, _, files in os.walk(search_root):
                if 'train.csv' in files: train_path = os.path.join(root, 'train.csv')
                if 'test.csv' in files: test_path = os.path.join(root, 'test.csv')
    print(f'  Data paths: {train_path}')
    return train_path, test_path

train_path, test_path = locate_datasets()
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(f'Training: {len(train_df)} samples, Test: {len(test_df)} samples')

def create_features(df):
    result = df.copy()
    dist = result['dist_min_ci_0_5h'].clip(lower=1)
    speed = result['closing_speed_m_per_h']
    perimeters = result['num_perimeters_0_5h']
    area_first = result['area_first_ha']
    result['log_distance'] = np.log1p(dist)
    result['inv_distance'] = 1 / (dist / 1000 + 0.1)
    result['inv_distance_sq'] = result['inv_distance'] ** 2
    result['sqrt_distance'] = np.sqrt(dist)
    result['dist_km'] = dist / 1000
    result['dist_km_sq'] = (dist / 1000) ** 2
    result['dist_rank'] = dist.rank(pct=True)
    fire_radius = np.sqrt(area_first * 10000 / np.pi)
    result['radius_to_dist'] = fire_radius / dist
    result['area_to_dist_ratio'] = area_first / (dist / 1000 + 0.1)
    result['log_area_dist_ratio'] = np.log1p(area_first) - np.log1p(dist)
    result['has_movement'] = (perimeters > 1).astype(float)
    closing_pos = speed.clip(lower=0)
    result['eta_hours'] = np.where(closing_pos > 0.01, dist / closing_pos, 9999).clip(max=9999)
    result['log_eta'] = np.log1p(result['eta_hours'].clip(0, 9999))
    radial_growth = result['radial_growth_rate_m_per_h'].clip(lower=0)
    effective_closing = closing_pos + radial_growth
    result['effective_closing_speed'] = effective_closing
    result['eta_effective'] = np.where(effective_closing > 0.01, dist / effective_closing, 9999).clip(max=9999)
    result['threat_score'] = result['alignment_abs'] * speed / np.log1p(dist)
    result['fire_urgency'] = perimeters * speed
    result['growth_intensity'] = result['area_growth_rate_ha_per_h'] * perimeters
    result['zone_critical'] = (dist < 5000).astype(float)
    result['zone_warning'] = ((dist >= 5000) & (dist < 10000)).astype(float)
    result['zone_safe'] = (dist >= 10000).astype(float)
    result['is_summer'] = result['event_start_month'].isin([6, 7, 8]).astype(float)
    result['is_afternoon'] = ((result['event_start_hour'] >= 12) & (result['event_start_hour'] < 20)).astype(float)
    drop_cols = ['relative_growth_0_5h', 'projected_advance_m', 'centroid_displacement_m',
                 'centroid_speed_m_per_h', 'closing_speed_abs_m_per_h', 'area_growth_abs_0_5h']
    result = result.drop(columns=[c for c in drop_cols if c in result.columns])
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result

train_processed = create_features(train_df)
test_processed = create_features(test_df)
def get_surv_predictions(model, X):
    surv_fns = model.predict_survival_function(X)
    preds = np.empty((len(surv_fns), len(HORIZONS_PRED)), dtype=float)
    for i, fn in enumerate(surv_fns):
        t_min, t_max = fn.domain
        preds[i, :] = fn(np.clip(HORIZONS_PRED, t_min, t_max))
    return 1.0 - preds

def sigmoid_pred(dist, threshold, scale):
    return 1.0 / (1.0 + np.exp((dist - threshold) / scale))

def make_binary_target(time_vals, event_vals, horizon):
    unknown = (event_vals == 0) & (time_vals < horizon)
    y = ((event_vals == 1) & (time_vals <= horizon)).astype(float)
    return y, ~unknown

def compute_ipcw_weights(times, events, horizon):
    unique_t = np.sort(np.unique(times))
    surv = np.ones(len(unique_t))
    for i, t in enumerate(unique_t):
        at_risk = (times >= t).sum()
        censored_at_t = ((times == t) & (events == 0)).sum()
        if at_risk > 0: surv[i] = 1 - censored_at_t / at_risk
        if i > 0: surv[i] *= surv[i - 1]
    def G(t):
        idx = np.searchsorted(unique_t, t, side='right') - 1
        return max(surv[idx], 0.01) if idx >= 0 else 1.0
    weights = np.ones(len(times))
    for i in range(len(times)):
        if events[i] == 1 and times[i] <= horizon: weights[i] = 1.0 / G(times[i])
        elif times[i] >= horizon: weights[i] = 1.0 / G(horizon)
    return weights

def enforce_monotonicity(preds):
    result = np.clip(preds, 0, 1)
    for i in range(1, result.shape[1]):
        result[:, i] = np.maximum(result[:, i], result[:, i-1])
    return result
X_surv_train = train_df.drop(columns=['event_id', 'event', 'time_to_hit_hours'])
X_surv_test = test_df.drop(columns=['event_id'])
y_surv = Surv.from_arrays(event=train_df['event'].astype(bool), time=train_df['time_to_hit_hours'])
event_values = train_df['event'].values
time_values = train_df['time_to_hit_hours'].values
dist_train = train_df['dist_min_ci_0_5h'].values
dist_test = test_df['dist_min_ci_0_5h'].values

# ═══════════════════════════════════════════════════════
# Phase 1: GBSA Ensemble (5 configs × 15 seeds × 5-fold)
# ═══════════════════════════════════════════════════════
print('\n═══ Phase 1: GBSA Ensemble ═══')
gbsa_configs = [
    {'learning_rate': 0.01, 'subsample': 0.7,  'max_depth': 3, 'min_samples_leaf': 12, 'min_samples_split': 3, 'n_estimators': 1200, 'dropout_rate': 0.0},
    {'learning_rate': 0.01, 'subsample': 0.85, 'max_depth': 3, 'min_samples_leaf': 15, 'min_samples_split': 3, 'n_estimators': 1200, 'dropout_rate': 0.0},
    {'learning_rate': 0.01, 'subsample': 0.6,  'max_depth': 3, 'min_samples_leaf': 12, 'min_samples_split': 3, 'n_estimators': 1200, 'dropout_rate': 0.0},
    {'learning_rate': 0.005,'subsample': 0.85, 'max_depth': 3, 'min_samples_leaf': 12, 'min_samples_split': 3, 'n_estimators': 2000, 'dropout_rate': 0.0},
    {'learning_rate': 0.01, 'subsample': 0.85, 'max_depth': 3, 'min_samples_leaf': 20, 'min_samples_split': 3, 'n_estimators': 1400, 'dropout_rate': 0.0},
]
N_SEEDS = 15

oof_gbsa = np.zeros((len(X_surv_train), 4))
test_gbsa = np.zeros((len(X_surv_test), 4))
total_models = len(gbsa_configs) * N_SEEDS * 5
model_count = 0
t_start = timer.time()

for cfg_idx, cfg in enumerate(gbsa_configs):
    cfg_oof = np.zeros((len(X_surv_train), 4))
    cfg_test = np.zeros((len(X_surv_test), 4))
    for seed in range(42, 42 + N_SEEDS):
        seed_test = np.zeros((len(X_surv_test), 4))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_surv_train, event_values)):
            m = GradientBoostingSurvivalAnalysis(**{**cfg, 'random_state': seed})
            m.fit(X_surv_train.iloc[tr_idx], y_surv[tr_idx])
            cfg_oof[va_idx] += get_surv_predictions(m, X_surv_train.iloc[va_idx]) / N_SEEDS
            seed_test += get_surv_predictions(m, X_surv_test) / 5
            model_count += 1
        cfg_test += seed_test / N_SEEDS
    oof_gbsa += cfg_oof / len(gbsa_configs)
    test_gbsa += cfg_test / len(gbsa_configs)
    elapsed = timer.time() - t_start
    print(f'  GBSA {cfg_idx+1}/{len(gbsa_configs)} done [{model_count}/{total_models}, {elapsed/60:.1f}m]')
print(f'GBSA done: {total_models} fold models')

# ═══════════════════════════════════════════════════════
# Phase 1b: ExtraSurvivalTrees (2 configs × 10 seeds)
# ═══════════════════════════════════════════════════════
print('\n═══ Phase 1b: ExtraSurvivalTrees ═══')
est_configs = [
    {'n_estimators': 500, 'max_depth': 5, 'min_samples_leaf': 8, 'min_samples_split': 3, 'max_features': 'sqrt'},
    {'n_estimators': 800, 'max_depth': 4, 'min_samples_leaf': 12, 'min_samples_split': 3, 'max_features': 0.7},
]
N_EST_SEEDS = 10
oof_est = np.zeros((len(X_surv_train), 4))
test_est = np.zeros((len(X_surv_test), 4))

for cfg_idx, cfg in enumerate(est_configs):
    cfg_oof = np.zeros((len(X_surv_train), 4))
    cfg_test = np.zeros((len(X_surv_test), 4))
    for seed in range(42, 42 + N_EST_SEEDS):
        seed_test = np.zeros((len(X_surv_test), 4))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, va_idx in cv.split(X_surv_train, event_values):
            m = ExtraSurvivalTrees(**{**cfg, 'random_state': seed, 'n_jobs': -1})
            m.fit(X_surv_train.iloc[tr_idx], y_surv[tr_idx])
            cfg_oof[va_idx] += get_surv_predictions(m, X_surv_train.iloc[va_idx]) / N_EST_SEEDS
            seed_test += get_surv_predictions(m, X_surv_test) / 5
        cfg_test += seed_test / N_EST_SEEDS
    oof_est += cfg_oof / len(est_configs)
    test_est += cfg_test / len(est_configs)
    print(f'  EST config {cfg_idx+1}/{len(est_configs)} done')
print(f'EST done: {len(est_configs) * N_EST_SEEDS * 5} fold models')

# ═══════════════════════════════════════════════════════
# Phase 1c: RandomSurvivalForest (1 config × 10 seeds)
# ═══════════════════════════════════════════════════════
print('\n═══ Phase 1c: RandomSurvivalForest ═══')
N_RSF_SEEDS = 10
oof_rsf = np.zeros((len(X_surv_train), 4))
test_rsf = np.zeros((len(X_surv_test), 4))
for seed in range(42, 42 + N_RSF_SEEDS):
    seed_test = np.zeros((len(X_surv_test), 4))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in cv.split(X_surv_train, event_values):
        m = RandomSurvivalForest(n_estimators=500, max_depth=5, min_samples_leaf=10,
                                 max_features='sqrt', random_state=seed, n_jobs=-1)
        m.fit(X_surv_train.iloc[tr_idx], y_surv[tr_idx])
        oof_rsf[va_idx] += get_surv_predictions(m, X_surv_train.iloc[va_idx]) / N_RSF_SEEDS
        seed_test += get_surv_predictions(m, X_surv_test) / 5
    test_rsf += seed_test / N_RSF_SEEDS
print(f'RSF done: {N_RSF_SEEDS * 5} fold models')

# ═══════════════════════════════════════════════════════
# Phase 2: CoxPH + Coxnet (linear survival models)
# ═══════════════════════════════════════════════════════
print('\n═══ Phase 2: CoxPH + Coxnet ═══')
scaler = StandardScaler()
X_cox_train = scaler.fit_transform(X_surv_train)
X_cox_test  = scaler.transform(X_surv_test)

oof_cox = np.zeros((len(X_surv_train), 4))
test_cox = np.zeros((len(X_surv_test), 4))
N_COX_SEEDS = 10
for seed in range(42, 42 + N_COX_SEEDS):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    seed_test = np.zeros((len(X_surv_test), 4))
    for tr_idx, va_idx in cv.split(X_cox_train, event_values):
        try:
            cph = CoxPHSurvivalAnalysis(alpha=0.1)
            cph.fit(X_cox_train[tr_idx], y_surv[tr_idx])
            oof_cox[va_idx] += get_surv_predictions(cph, X_cox_train[va_idx]) / N_COX_SEEDS
            seed_test += get_surv_predictions(cph, X_cox_test) / 5
        except Exception:
            seed_test += test_gbsa / 5
    test_cox += seed_test / N_COX_SEEDS
print(f'  CoxPH done ({N_COX_SEEDS} seeds x 5-fold)')

oof_coxnet = np.zeros((len(X_surv_train), 4))
test_coxnet = np.zeros((len(X_surv_test), 4))
N_COXNET_SEEDS = 10
for seed in range(42, 42 + N_COXNET_SEEDS):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    seed_test = np.zeros((len(X_surv_test), 4))
    for tr_idx, va_idx in cv.split(X_cox_train, event_values):
        try:
            cnet = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.1, n_alphas=50)
            cnet.fit(X_cox_train[tr_idx], y_surv[tr_idx])
            oof_coxnet[va_idx] += get_surv_predictions(cnet, X_cox_train[va_idx]) / N_COXNET_SEEDS
            seed_test += get_surv_predictions(cnet, X_cox_test) / 5
        except Exception:
            seed_test += test_gbsa / 5
    test_coxnet += seed_test / N_COXNET_SEEDS
print(f'  Coxnet done ({N_COXNET_SEEDS} seeds x 5-fold)')

# ═══════════════════════════════════════════════════════
# OOF-Optimized Survival Blend Weights
# ═══════════════════════════════════════════════════════
print('\n═══ Optimizing Survival Blend Weights (OOF) ═══')
def surv_blend_loss(w):
    w = np.abs(w)
    w = w / w.sum()
    blended = w[0]*oof_gbsa + w[1]*oof_est + w[2]*oof_rsf + w[3]*oof_cox + w[4]*oof_coxnet
    total_bs = 0
    for h_idx, h in enumerate([12, 24, 48, 72]):
        y_bin, mask = make_binary_target(time_values, event_values, h)
        valid = np.where(mask)[0]
        total_bs += brier_score_loss(y_bin[valid], np.clip(blended[valid, h_idx], 0, 1))
    return total_bs

res = minimize(surv_blend_loss, x0=[0.6, 0.1, 0.1, 0.1, 0.1],
               method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6})
w_opt = np.abs(res.x); w_opt = w_opt / w_opt.sum()
print(f'  Optimal weights: GBSA={w_opt[0]:.3f}, EST={w_opt[1]:.3f}, RSF={w_opt[2]:.3f}, Cox={w_opt[3]:.3f}, Coxnet={w_opt[4]:.3f}')
print(f'  OOF Brier (optimized): {res.fun:.5f}')

oof_surv = w_opt[0]*oof_gbsa + w_opt[1]*oof_est + w_opt[2]*oof_rsf + w_opt[3]*oof_cox + w_opt[4]*oof_coxnet
test_surv = w_opt[0]*test_gbsa + w_opt[1]*test_est + w_opt[2]*test_rsf + w_opt[3]*test_cox + w_opt[4]*test_coxnet

# ═══════════════════════════════════════════════════════
# PowerCal on OOF (grid search best alpha per horizon)
# ═══════════════════════════════════════════════════════
print('\n═══ PowerCal Grid Search ═══')
best_alphas = []
for h_idx, h in enumerate([12, 24, 48, 72]):
    y_bin, mask = make_binary_target(time_values, event_values, h)
    valid = np.where(mask)[0]
    best_a, best_bs = 1.0, 999
    for a in np.arange(0.80, 1.50, 0.02):
        calibrated = np.clip(oof_surv[valid, h_idx] ** a, 0, 1)
        bs = brier_score_loss(y_bin[valid], calibrated)
        if bs < best_bs: best_a, best_bs = a, bs
    best_alphas.append(best_a)
    print(f'  {h}h: alpha={best_a:.2f}, Brier={best_bs:.5f}')

# Apply PowerCal
for h_idx in range(4):
    oof_surv[:, h_idx] = np.clip(oof_surv[:, h_idx] ** best_alphas[h_idx], 0, 1)
    test_surv[:, h_idx] = np.clip(test_surv[:, h_idx] ** best_alphas[h_idx], 0, 1)
print('  PowerCal applied to all horizons')


# ═══════════════════════════════════════════════════════
# Phase 3: LGB IPCW Classifiers (12h, 24h, 48h)
# ═══════════════════════════════════════════════════════
print('\n═══ Phase 3: LGB IPCW Classifiers ═══')
X_lgb_train = train_processed.drop(columns=['event_id', 'event', 'time_to_hit_hours'])
X_lgb_test = test_processed.drop(columns=['event_id'])

lgb_cfgs = {
    12: {'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 250,
         'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_samples': 10,
         'reg_alpha': 0.5, 'reg_lambda': 2.0, 'num_leaves': 7},
    24: {'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 300,
         'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_samples': 8,
         'reg_alpha': 0.5, 'reg_lambda': 2.0, 'num_leaves': 7},
    48: {'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 200,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 5,
         'reg_alpha': 0.1, 'reg_lambda': 1.0, 'num_leaves': 4},
}
N_LGB_SEEDS = 20
lgb_oof = {}
lgb_test = {}

for horizon in [12, 24, 48]:
    y_bin, mask = make_binary_target(time_values, event_values, horizon)
    valid_idx = np.where(mask)[0]
    cfg = lgb_cfgs[horizon]
    all_oof = np.zeros(len(X_lgb_train))
    all_test = np.zeros(len(X_lgb_test))
    for seed in range(42, 42 + N_LGB_SEEDS):
        seed_test = np.zeros(len(X_lgb_test))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_v, va_v in cv.split(valid_idx, y_bin[mask]):
            tr_idx = valid_idx[tr_v]
            va_idx = valid_idx[va_v]
            ipcw_w = compute_ipcw_weights(time_values[tr_idx], event_values[tr_idx], horizon)
            m = lgb.LGBMClassifier(**cfg, objective='binary', random_state=seed, verbose=-1)
            m.fit(X_lgb_train.iloc[tr_idx], y_bin[tr_idx], sample_weight=ipcw_w)
            all_oof[va_idx] += m.predict_proba(X_lgb_train.iloc[va_idx])[:, 1] / N_LGB_SEEDS
            seed_test += m.predict_proba(X_lgb_test)[:, 1] / 5
        all_test += seed_test
    lgb_oof[horizon] = all_oof
    lgb_test[horizon] = all_test / N_LGB_SEEDS
    print(f'  LGB {horizon}h IPCW done')

# ═══════════════════════════════════════════════════════
# OOF-Optimized Surv-LGB Blend Weights per Horizon
# ═══════════════════════════════════════════════════════
print('\n═══ Optimizing Surv-LGB Blend ═══')
best_W = {}
for horizon in [12, 24, 48]:
    h_idx = [12, 24, 48].index(horizon)
    y_bin, mask = make_binary_target(time_values, event_values, horizon)
    valid = np.where(mask)[0]
    best_w, best_bs = 0.5, 999
    for w in np.arange(0.0, 1.01, 0.01):
        blended = w * oof_surv[valid, h_idx] + (1 - w) * lgb_oof[horizon][valid]
        bs = brier_score_loss(y_bin[valid], np.clip(blended, 0, 1))
        if bs < best_bs: best_w, best_bs = w, bs
    best_W[horizon] = best_w
    print(f'  {horizon}h: W_surv={best_w:.2f}, Brier={best_bs:.5f}')

# ═══════════════════════════════════════════════════════
# Final Assembly (Direct Blend + Hard 5km Cutoff)
# ═══════════════════════════════════════════════════════
print('\n═══ Final Assembly ═══')

# Direct blend with OOF-optimized weights
oof_blend = oof_surv.copy()
oof_blend[:, 0] = best_W[12] * oof_surv[:, 0] + (1 - best_W[12]) * lgb_oof[12]
oof_blend[:, 1] = best_W[24] * oof_surv[:, 1] + (1 - best_W[24]) * lgb_oof[24]
oof_blend[:, 2] = best_W[48] * oof_surv[:, 2] + (1 - best_W[48]) * lgb_oof[48]
oof_blend[:, 3] = sigmoid_pred(dist_train, 5450, 50)

test_blend = test_surv.copy()
test_blend[:, 0] = best_W[12] * test_surv[:, 0] + (1 - best_W[12]) * lgb_test[12]
test_blend[:, 1] = best_W[24] * test_surv[:, 1] + (1 - best_W[24]) * lgb_test[24]
test_blend[:, 2] = best_W[48] * test_surv[:, 2] + (1 - best_W[48]) * lgb_test[48]
test_blend[:, 3] = sigmoid_pred(dist_test, 5450, 50)

# Hard 5km cutoff (PROVEN - do not change)
far_mask_test = dist_test >= 5000
test_blend[far_mask_test, :] = 0.0
far_mask_train = dist_train >= 5000
oof_blend[far_mask_train, :] = 0.0

test_final = enforce_monotonicity(test_blend)
oof_final = enforce_monotonicity(oof_blend)

# ═══════════════════════════════════════════════════════
# OOF SCORING
# ═══════════════════════════════════════════════════════
print('\n' + '='*55)
print('  OOF PERFORMANCE REPORT (Level 7)')
print('='*55)
for h_idx, h in enumerate([12, 24, 48, 72]):
    y_bin, mask = make_binary_target(time_values, event_values, h)
    valid = np.where(mask)[0]
    bs = brier_score_loss(y_bin[valid], oof_final[valid, h_idx])
    print(f'  {h}h Brier Score: {bs:.5f}')
weighted_brier = (
    0.3 * brier_score_loss(*[v[np.where(make_binary_target(time_values, event_values, 24)[1])[0]] for v in [make_binary_target(time_values, event_values, 24)[0], oof_final[:, 1]]]) +
    0.3 * brier_score_loss(*[v[np.where(make_binary_target(time_values, event_values, 48)[1])[0]] for v in [make_binary_target(time_values, event_values, 48)[0], oof_final[:, 2]]]) +
    0.4 * brier_score_loss(*[v[np.where(make_binary_target(time_values, event_values, 72)[1])[0]] for v in [make_binary_target(time_values, event_values, 72)[0], oof_final[:, 3]]])
)
print(f'\n  Weighted Brier: {weighted_brier:.5f}')
print(f'  Hybrid Score:   {0.5 * (1 - weighted_brier) + 0.5:.5f}  (approx)')
print('='*55)

submission = pd.DataFrame({
    'event_id': test_df['event_id'],
    'prob_12h': test_final[:, 0],
    'prob_24h': test_final[:, 1],
    'prob_48h': test_final[:, 2],
    'prob_72h': test_final[:, 3],
})

output_path = '/kaggle/working/submission.csv' if os.path.isdir('/kaggle/working') else 'submissionpapi.csv'
submission.to_csv(output_path, index=False)
print(f'\nSaved: {output_path}')
print(submission.describe().round(4).to_string())