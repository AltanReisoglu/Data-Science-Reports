import os
import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import Counter

# ---------------------------------------------------------
# 1. PARAMETRELER VE MODEL TANIMI
# ---------------------------------------------------------
BATCH_SIZE = 128
PRETRAIN_ITERS = 200  # Tüm 10 rakam — kasıtlı kısa ("orta halli" base model)
N = 1000  # RandOpt Popülasyonu (Perturbasyon Sayısı)
K = 5     # Ensemble Seçilecek Model Sayısı
GLOBAL_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # 28x28 resimler -> 784 girdi
        self.features = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10) # 10 Sınıf (0-9)
        )

    def forward(self, x):
        x = x.view(-1, 784) # Flatten
        return self.features(x)

    def perturb_weights(self, seed, sigma):
        """RandOpt'un çekirdeği: Klasik gradyan yerine reproducible rastgele gürültü."""
        torch.manual_seed(seed)
        for p in self.parameters():
            p.data.add_(torch.randn_like(p.data) * sigma)

# ---------------------------------------------------------
# 2. VERİ YÜKLERİ
# ---------------------------------------------------------
def get_loader(train=True, bsz=BATCH_SIZE):
    """Tüm 10 rakamı (0-9) içeren standart MNIST loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=train)
    return loader

# ---------------------------------------------------------
# 3. PRE-TRAINING (0-4 Digits)
# ---------------------------------------------------------
def pretrain_base_model(model, loader, iters=500, lr=0.001):
    print(f"🎬 [Faza 1] Base Model Rakamlar [0-9] ile Eğitiliyor ({iters} iterasyon)...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    data_iter = iter(loader)
    
    for i in range(iters):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, labels = next(data_iter)
            
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f"  Iter {i+1}/{iters} | Loss: {loss.item():.4f}")
            
    print("✅ Base Model Eğitimi Bitti.")
    return model

# ---------------------------------------------------------
# 4. RANDOPT SEÇİM MEKANİZMASI (5-9 Digits)
# ---------------------------------------------------------
def randopt(base_model, loader, N=N, sigma=0.01, K=K):
    """Ağırlıklara gürültü katarak yeni rakamlar (5-9) üzerinde en doğru K seed'i bulur."""
    print(f"\n🎲 [Faza 2] RandOpt Başlatılıyor: {N} Perturbasyon Arama Yapılıyor (Sigma={sigma})...")
    
    # RandOpt optimizasyonunu küçük bir "Hizalama/Kalibrasyon" batch'inde yapıyoruz.
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    
    model_scores = []
    t0 = time.time()
    
    for seed in range(N):
        # Base modeli kopyala ve ağırlığını boz
        perturbed = copy.deepcopy(base_model)
        perturbed.perturb_weights(seed, sigma)
        perturbed.eval()
        
        with torch.no_grad():
            outputs = perturbed(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)
            
        model_scores.append((seed, accuracy))
        
    # Doğruluğa göre BÜYÜKTEN KÜÇÜĞE Sırala
    model_scores.sort(key=lambda x: x[1], reverse=True)
    best_seeds = [model_scores[i][0] for i in range(K)]
    
    print(f"⏱ RandOpt Done in {time.time() - t0:.1f}s")
    print(f"📈 En İyi Seed Doğruluk Oranı: %{model_scores[0][1]*100:.2f}")
    return best_seeds

# ---------------------------------------------------------
# 5. ENSEMBLE TEST VE ÇOĞUNLUK OYLAMASI
# ---------------------------------------------------------
def evaluate_and_plot(base_model, top_k_seeds, sigma, loader, K=5):
    print("\n📊 [Faza 3] Sonuçlar Test Ediliyor ve Çoğunluk Oylaması Yapılıyor...")
    base_model.eval()
    
    total_samples = 0
    base_correct = 0
    ensemble_correct = 0
    
    # En iyi K modeli RAM'e yükle (Maliyetli değildir çünkü seed bazlıdır)
    ensemble_models = []
    for seed in top_k_seeds:
        m = copy.deepcopy(base_model)
        m.perturb_weights(seed, sigma)
        m.eval()
        ensemble_models.append(m)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            total_samples += labels.size(0)
            
            # 1. Base Model Tahmini
            base_out = base_model(images)
            _, base_pred = torch.max(base_out, 1)
            base_correct += (base_pred == labels).sum().item()
            
            # 2. Ensemble (Çoğunluk Oylaması)
            # Her bir modelin tekil resimler için tahminlerini al
            batch_predictions = []
            for m in ensemble_models:
                out = m(images)
                _, pred = torch.max(out, 1)
                batch_predictions.append(pred.cpu().numpy())
                
            # numpy dizisine çevir: (model_sayisi, batch_size)
            batch_predictions = np.stack(batch_predictions)
            
            # Her bir resim (sütun) için en çok oy alan sınıfı bul
            for i in range(labels.size(0)):
                answers = batch_predictions[:, i]
                majority_vote = Counter(answers).most_common(1)[0][0]
                if majority_vote == labels[i].cpu().item():
                    ensemble_correct += 1

    accuracy_base = base_correct / total_samples
    accuracy_ens = ensemble_correct / total_samples
    
    print("="*40)
    print(f"🔴 Base Model [0-9 Tahmin] Başarısı  : %{accuracy_base*100:.2f}")
    print(f"🟡 RandOpt Ensemble [0-9 Tahmin] Başarısı: %{accuracy_ens*100:.2f}")
    print("="*40)

    # --- Bar Chart ---
    plt.figure(figsize=(6, 5))
    categories = ['Base (0-9 Eğitilmiş)', 'RandOpt Ensemble (0-9)']
    values = [accuracy_base * 100, accuracy_ens * 100]
    
    bars = plt.bar(categories, values, color=['#E64B35', '#F39C12'])
    plt.ylabel('Doğruluk Oranı (%)')
    plt.title('MNIST Adaptasyon: Klasik vs RandOpt (Neural Thickets)')
    plt.ylim(0, 100)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'%{yval:.1f}', ha='center', va='bottom', fontweight='bold')
        
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/randopt_mnist_bar.png")
    print("📊 Grafik kaydedildi: results/randopt_mnist_bar.png")
    plt.show()

# ---------------------------------------------------------
# RUNNER
# ---------------------------------------------------------
def main():
    os.makedirs("data", exist_ok=True)
    torch.manual_seed(GLOBAL_SEED)
    
    # 1. Base Model
    model = MNISTNet().to(device)
    
    # 2. Tüm 10 rakam — kısa pretraining (kasıtlı "orta halli")
    train_loader = get_loader(train=True, bsz=BATCH_SIZE)
    pretrain_base_model(model, train_loader, iters=PRETRAIN_ITERS)
    
    # 3. Kalibrasyon batch'i (makaledeki küçük D_train)
    calib_loader = get_loader(train=True, bsz=256)
    
    # 4. Tam test seti
    test_loader = get_loader(train=False, bsz=256)
    
    # 5. RandOpt (sigma=0.005 — makaledeki değer)
    sigma = 0.005
    top_k_seeds = randopt(model, calib_loader, N=N, sigma=sigma, K=K)
    
    # 6. Değerlendirme
    evaluate_and_plot(model, top_k_seeds, sigma, test_loader, K=K)

if __name__ == "__main__":
    main()

