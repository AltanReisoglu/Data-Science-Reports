"""
KAPSAMLI GORSEL ANALIZ - Boya Fabrikasi Datathonu
Her komut tipi icin bk_level dinamigini gorsellestir
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

print("Veri yukleniyor...")
train = pd.read_csv("aiclubdatathon-26/train.csv")
test  = pd.read_csv("aiclubdatathon-26/test.csv")

for df in [train, test]:
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df["starttime"] = pd.to_datetime(df["starttime"], format="mixed", utc=True)
    df["endtime"]   = pd.to_datetime(df["endtime"],   format="mixed", utc=True)
    for c in df.select_dtypes(include="bool").columns:
        df[c] = df[c].astype(int)
    df["process_id"] = (df["machineid"].astype(str) + "_" +
                        df["batchkey"].astype(str) + "_" + df["commandno"].astype(str))

train = train.sort_values(["process_id","timestamp"]).reset_index(drop=True)
test  = test.sort_values(["process_id","timestamp"]).reset_index(drop=True)

dur = (train["endtime"] - train["starttime"]).dt.total_seconds().clip(lower=1)
ela = (train["timestamp"] - train["starttime"]).dt.total_seconds()
train["progress"] = (ela / dur).clip(0, 1)

train_nz = train[train["bk_level"] > 0].copy()

VALVE_COLS = ["fast_dosage_valve","slow_dosage_valve","kk_dosage_valve",
              "bk_dosage_valve","kk_bk_common_discharge","bk_irtibat_valve","kk_irtibat_valve"]

# ─────────────────────────────────────────────────────────
# FIGURE 1: Her komut icin tipik bk_level profili
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("bk_level Profili - Komut Tipine Gore (Process Progress)", fontsize=16, fontweight="bold")

cmd_labels = {19: "Cmd 19: KK->AK Transfer", 20: "Cmd 20: BK->AK Transfer",
              21: "Cmd 21: KK->AK Dozaj",    22: "Cmd 22: BK->AK Dozaj"}
axes_flat = axes.flat

for ax, cmd in zip(axes_flat, [19, 20, 21, 22]):
    sub = train_nz[train_nz["commandno"] == cmd]
    # Her prosesi progress eksenine normalize et, 5 ornek proses
    pids = sub.groupby("process_id").size()
    pids = pids[pids > 20].index[:8]
    
    for pid in pids:
        p = sub[sub["process_id"] == pid].sort_values("progress")
        ax.plot(p["progress"], p["bk_level"], alpha=0.5, linewidth=1)
    
    # Binned ortalama
    sub2 = sub.copy()
    sub2["prog_bin"] = pd.cut(sub2["progress"], bins=50, labels=False)
    binned = sub2.groupby("prog_bin")["bk_level"].agg(["mean","std"]).dropna()
    bins_x = (binned.index.astype(float) + 0.5) / 50  # center of each bin
    ax.plot(bins_x, binned["mean"].values, "k--", linewidth=2.5, label="Ortalama")
    ax.fill_between(bins_x, 
                    (binned["mean"] - binned["std"]).values, 
                    (binned["mean"] + binned["std"]).values,
                    alpha=0.15, color="black")
    
    # bk_target_level ortalama
    tgt_mean = sub["bk_target_level"].mean()
    ax.axhline(tgt_mean, color="red", linestyle=":", linewidth=1.5, label=f"bk_target mean={tgt_mean:.1f}")
    
    ax.set_title(cmd_labels[cmd], fontsize=12, fontweight="bold")
    ax.set_xlabel("Process Progress (0=start, 1=end)")
    ax.set_ylabel("bk_level (%)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analysis_bk_profiles.png", dpi=120, bbox_inches="tight")
plt.close()
print("Kaydedildi: analysis_bk_profiles.png")

# ─────────────────────────────────────────────────────────
# FIGURE 2: Korelasyon analizi - hangi feature'lar onemli?
# ─────────────────────────────────────────────────────────
for v in VALVE_COLS:
    train_nz[f"cum_{v}"] = train_nz.groupby("process_id")[v].cumsum()
train_nz["cum_total"] = train_nz[VALVE_COLS].sum(axis=1)
train_nz["cum_total_int"] = train_nz.groupby("process_id")["cum_total"].cumsum()

# Korelasyonlar
feat_cols = ["progress", "bk_target_level", "kk_level", "ak_level", 
             "cum_fast_dosage_valve", "cum_bk_dosage_valve",
             "cum_bk_irtibat_valve", "cum_total_int",
             "bk_irtibat_valve", "fast_dosage_valve", "slow_dosage_valve"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, cmd_group, title in zip(axes, [[20,22],[19,21]], 
                                  ["BK Dreni (20,22)", "KK Dreni (19,21)"]):
    sub = train_nz[train_nz["commandno"].isin(cmd_group)].sample(min(50000, len(train_nz)), random_state=42)
    corr = sub[[c for c in feat_cols if c in sub.columns] + ["bk_level"]].corr()["bk_level"].drop("bk_level")
    corr_sorted = corr.abs().sort_values(ascending=True)
    
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in corr[corr_sorted.index].values]
    ax.barh(corr_sorted.index, corr_sorted.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(f"bk_level Korelasyonu - {title}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mutlak Korelasyon")
    ax.axvline(0.1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("analysis_correlations.png", dpi=120, bbox_inches="tight")
plt.close()
print("Kaydedildi: analysis_correlations.png")

# ─────────────────────────────────────────────────────────
# FIGURE 3: Makine bazli bk_level dagilimi
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Makine bazli
machine_stats = train_nz.groupby(["machineid","commandno"])["bk_level"].mean().unstack()
machine_stats.plot(kind="bar", ax=axes[0], colormap="Set2", edgecolor="white")
axes[0].set_title("Makine x Komut Bazli Ortalama bk_level", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Makine ID")
axes[0].set_ylabel("Ortalama bk_level (%)")
axes[0].legend(title="Komut", loc="upper right")
axes[0].grid(True, alpha=0.3, axis="y")
axes[0].tick_params(axis="x", rotation=0)

# baslangic vs bitis dagilim
first_bk = train_nz.groupby(["process_id","commandno"])["bk_level"].first().reset_index()
last_bk  = train_nz.groupby(["process_id","commandno"])["bk_level"].last().reset_index()

cmd_colors = {19:"#3498db", 20:"#e74c3c", 21:"#2ecc71", 22:"#e67e22"}
for cmd in [19,20,21,22]:
    f = first_bk[first_bk["commandno"]==cmd]["bk_level"]
    axes[1].hist(f, bins=50, alpha=0.5, label=f"Cmd {cmd} (n={len(f)})", 
                 color=cmd_colors[cmd], density=True)

axes[1].set_title("Proses Baslangic bk_level Dagilimi", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Baslangic bk_level (%)")
axes[1].set_ylabel("Yogunluk")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analysis_distributions.png", dpi=120, bbox_inches="tight")
plt.close()
print("Kaydedildi: analysis_distributions.png")

# ─────────────────────────────────────────────────────────
# FIGURE 4: Ayni batch'in train/test'te olmasi
# ─────────────────────────────────────────────────────────
train["machine_batch"] = train["machineid"].astype(str) + "_" + train["batchkey"].astype(str)
test["machine_batch"]  = test["machineid"].astype(str)  + "_" + test["batchkey"].astype(str)

common_batches = set(train["machine_batch"]) & set(test["machine_batch"])
print(f"\nAyni batch'in train ve test'te olmasi: {len(common_batches)} batch")
test_common = test[test["machine_batch"].isin(common_batches)]
test_only   = test[~test["machine_batch"].isin(common_batches)]
print(f"  Test satiri (ortak batch): {len(test_common):,} ({len(test_common)/len(test)*100:.1f}%)")
print(f"  Test satiri (sadece test): {len(test_only):,} ({len(test_only)/len(test)*100:.1f}%)")

# ─────────────────────────────────────────────────────────
# KONSOL OZETI  
# ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("ANALIZ OZETI")
print("="*55)

for cmd in [19, 20, 21, 22]:
    sub = train_nz[train_nz["commandno"] == cmd]
    first = sub.groupby("process_id")["bk_level"].first()
    last  = sub.groupby("process_id")["bk_level"].last()
    dur_s = sub.groupby("process_id").size()
    print(f"\nCmd {cmd}: {sub['process_id'].nunique()} proses, {len(sub):,} satir")
    print(f"  Baslangic bk:   mean={first.mean():.1f}  median={first.median():.1f}")
    print(f"  Bitis bk:       mean={last.mean():.1f}")
    print(f"  Delta:          mean={(last-first).mean():.2f}")
    print(f"  Proses uzunlugu: mean={dur_s.mean():.0f}s  max={dur_s.max()}s")
    tgt_corr = sub[["bk_level","bk_target_level"]].corr().iloc[0,1]
    print(f"  Korelasyon bk~target: {tgt_corr:.4f}")
    
print("\nTum gorseller kaydedildi!")
