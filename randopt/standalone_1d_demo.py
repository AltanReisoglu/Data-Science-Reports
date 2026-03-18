import os
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. PARAMETRELER VE MODEL TANIMI
# ---------------------------------------------------------
CTX_SZ = 10
FUT_SZ = 60
WIDTH = 128
DEPTH = 5
GLOBAL_SEED = 0
N = 1000  # Perturbasyon Popülasyonu
K = 5     # Ensemble Seçilecek Model Sayısı

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, width=WIDTH, depth=DEPTH, dim_in=CTX_SZ, dim_out=1, init_type="xavier"):
        super().__init__()
        layers = [nn.Linear(dim_in, width, device=device)]
        for _ in range(depth - 2):
            layers.extend([nn.ReLU(), nn.Linear(width, width, device=device)])
        layers.extend([nn.ReLU(), nn.Linear(width, dim_out, device=device)])
        self.layers = nn.ModuleList(layers)
        self.init_type = init_type

    def forward(self, ctx):
        was_1d = ctx.dim() == 1
        if was_1d:
            ctx = ctx.unsqueeze(0)
        h = ctx
        for layer in self.layers:
            h = layer(h)
        if was_1d:
            h = h.squeeze(0)
        return h.squeeze(-1)

    def compute_loss(self, ctx, y):
        y_pred = self.forward(ctx)
        return nn.MSELoss()(y_pred, y.squeeze(-1))

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight) if self.init_type == "xavier" else nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def perturb_weights(self, seed, sigma):
        """RandOpt'un çekirdeği: Ağırlıklara gürültü ekleme."""
        torch.manual_seed(seed)
        for p in self.parameters():
            p.data.add_(torch.randn_like(p.data) * sigma)

    def AR_rollout(self, ctx, T):
        """Autoregressively tahmin yapma."""
        y_preds = []
        for _ in range(T):
            y_pred = self.forward(ctx)
            ctx = torch.cat([ctx, y_pred.unsqueeze(-1)], dim=1)[:, 1:]
            y_preds.append(y_pred)
        return torch.stack(y_preds, dim=1)

# ---------------------------------------------------------
# 2. VERİ SETİ OLUŞTURUCU (SINUSOID)
# ---------------------------------------------------------
def generate_sinusoid():
    phase = np.random.uniform(0, 2 * np.pi)
    amp = np.random.uniform(0.8, 1.2)
    return lambda x: amp * np.sin(4.0 * np.asarray(x) + phase)

def load_data(bsz, ctx_sz, fut_sz):
    ctx_y_list, fut_y_list = [], []
    for _ in range(bsz):
        gt_fn = generate_sinusoid()
        x_vals = -2.5 + np.arange(ctx_sz + fut_sz) * 0.1
        y_vals = [float(gt_fn(x)) for x in x_vals]
        ctx_y_list.append(y_vals[:ctx_sz])
        fut_y_list.append(y_vals[ctx_sz:])
    return torch.tensor(ctx_y_list, dtype=torch.float32, device=device), torch.tensor(fut_y_list, dtype=torch.float32, device=device)

# ---------------------------------------------------------
# 3. RANDOPT - PARALEL SEÇİM VE ENSEMBLE
# ---------------------------------------------------------
def randopt(base_model, ctx_sz, fut_sz, N=N, sigma=0.05, K=K):
    print(f"🎬 RandOpt Başlatılıyor: {N} Perturbasyon Tahmin Ediliyor (Sigma={sigma})...")
    ctx_y, fut_y = load_data(128, ctx_sz, fut_sz)
    model_scores = []
    
    for seed in range(N):
        perturbed = copy.deepcopy(base_model)
        perturbed.perturb_weights(seed, sigma)
        perturbed.eval()
        with torch.no_grad():
            y_pred = perturbed.AR_rollout(ctx_y, fut_sz)
            # MSE Hesapla
            loss = ((y_pred - fut_y) ** 2).sum(dim=1).mean()
        model_scores.append((seed, loss.item()))
    
    model_scores.sort(key=lambda x: x[1])
    return [model_scores[i][0] for i in range(K)], model_scores[0][1]

# ---------------------------------------------------------
# 4. GÖRSELLEŞTİRME VE ÇIKTI MAKER
# ---------------------------------------------------------
def run():
    print("🚀 RandOpt (1D) Standalone Model Başlatılıyor... 🚀")
    os.makedirs("results", exist_ok=True)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    # 1. Base Model Oluştur
    base = Net()
    base.init_weights()

    # 2. RandOpt Çalıştır (Xavier -> Sinusoid)
    sigma = 0.05
    top_k_seeds, best_loss = randopt(base, CTX_SZ, FUT_SZ, N=N, sigma=sigma, K=K)
    print(f"✅ En İyi Tohum (Seed): {top_k_seeds[0]} - Loss: {best_loss:.4f}")

    # 3. Tahminleri Topla (Evaluate)
    ctx_y, fut_y = load_data(1, CTX_SZ, FUT_SZ)
    base.eval()
    with torch.no_grad():
        base_preds = base.AR_rollout(ctx_y, fut_sz)
        
        # En iyi K modelin ortalamasını al
        top_k_preds = []
        for seed in top_k_seeds:
            m = copy.deepcopy(base)
            m.perturb_weights(seed, sigma)
            m.eval()
            top_k_preds.append(m.AR_rollout(ctx_y, fut_sz))
        
        ensemble_preds = torch.stack(top_k_preds).mean(dim=0)

    # 4. Matplotlib ile Plotting
    plt.figure(figsize=(10, 5))
    x_axis = np.arange(FUT_SZ)
    
    # Ground Truth
    plt.plot(np.arange(-CTX_SZ, 0), ctx_y[0].cpu().numpy(), "b-", lw=3, label="Giriş / Context")
    plt.plot(x_axis, fut_y[0].cpu().numpy(), "b--", lw=3, label="Gerçek Veri (GT)")
    
    # Base Model (Fails)
    plt.plot(x_axis, base_preds[0].cpu().numpy(), "k-", lw=2, label="Base Model (Öğrenmemiş)")

    # Ensemble (RandOpt finds experts around)
    plt.plot(x_axis, ensemble_preds[0].cpu().numpy(), "r-", lw=3, label="RandOpt Ensemble (Top-K)")

    plt.title("RandOpt: 'Sinirsel Çalılıklar' 1D Kanıtı", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True)
    
    plot_path = "results/randopt_1d_visual_proof.png"
    plt.savefig(plot_path)
    print(f"📊 Görsel kanıt kaydedildi: {plot_path}")
    plt.show()

if __name__ == "__main__":
    run()
