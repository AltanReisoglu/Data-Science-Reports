import os
import sys
import argparse
import subprocess

def setup_randopt():
    """RandOpt kütüphanesini indirir ve python path'ine ekler."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.join(script_dir, "RandOpt_repo")
    
    if not os.path.exists(repo_dir):
        print("RandOpt GitHub reposu klonlanıyor...")
        subprocess.run(["git", "clone", "https://github.com/sunrainyg/RandOpt.git", repo_dir], check=True)
    
    # RandOpt modüllerini içe aktarabilmek için path'e ekliyoruz
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

def main():
    print("🚀 RandOpt Kütüphanesini Kullanarak Model Eğitimi Başlatılıyor 🚀")
    
    # 1. Framework hazırlığı (Klonlama ve Path ayarlama)
    setup_randopt()
    
    # 2. Artık RandOpt'u doğrudan bir Python modülü gibi kullanabiliriz
    import randopt
    
    # 3. Model Eğitimi için Argümanları Programatik Olarak Ayarlama
    # Normalde komut satırından verdiğimiz tüm argümanları burada python objesi olarak hazırlıyoruz.
    args = argparse.Namespace(
        dataset="gsm8k",  # GSM8K matematik veri seti
        train_data_path=None, 
        test_data_path=None,
        train_samples=50,  # Eğitim için kullanılacak soru sayısı (örnek için düşük tuttuk)
        test_samples=10,   # Test için kullanılacak soru sayısı
        model_name="Qwen/Qwen2.5-1.5B-Instruct", # Kullanılacak Base Model
        precision="bfloat16",
        max_tokens=256,
        sigma_values="0.0005,0.001", # Gürültü miktarları
        population_size=10, # Kaç farklı gürültülü model varyasyonu (perturbasyon) oluşturulacak
        top_k_ratios="0.1,0.2", # En iyi %10'u ve %20'si üzerinden çoğunluk oylaması yap
        num_engines=1, # GPU sayısı
        tp=1,          # Tensor Parallelism derecesi
        cuda_devices="0",
        global_seed=42,
        experiment_dir="python_randopt_experiment",
        resume_dir=None
    )
    
    # Gerekli ekstra array ve hesaplamaları randopt.py içerisindeki gibi ayarlıyoruz
    args.sigma_list = [float(s.strip()) for s in args.sigma_values.split(",")]
    ratios = [float(r.strip()) for r in args.top_k_ratios.split(",")]
    args.top_k_list = sorted(set(max(1, int(r * args.population_size)) for r in ratios), reverse=True)
    args.max_top_k = args.top_k_list[0]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    
    print(f"Hedef Model: {args.model_name}")
    print(f"Oluşturulacak Model Varyasyonu (Popülasyon): {args.population_size}")
    
    try:
        # RandOpt'un ana algoritmasını bash yerine doğrudan kendi kodumuzdan çağırıyoruz.
        # Bu fonksiyon; veri setini yükleyecek, Base Model'in performansını ölçecek,
        # ağırlıklara varyasyon katacak (Neural Thickets), ve en iyi modelleri seçecek.
        randopt.main(args)
        
        print("\n✅ Eğitim başarıyla tamamlandı! Sonuçlar 'python_randopt_experiment' klasöründedir.")
        
    except Exception as e:
        print(f"\n❌ Eğitim sırasında bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
