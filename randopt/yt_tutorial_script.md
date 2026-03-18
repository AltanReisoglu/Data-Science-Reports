# YouTube Tutorial: RandOpt (Neural Thickets) İle Model Eğitimi

## 🎬 Giriş (0:00 - 1:30)
**Görsel:** Ekranda "RandOpt: Neural Thickets" paper'ının başlığı ve GitHub reposu açık.
**Konuşma:**
"Herkese merhaba! Bugün yapay zeka dünyasında ezberleri bozan çok yeni ve çok ilginç bir çalışmadan bahsedeceğiz: **Neural Thickets** yani 'Sinirsel Çalılıklar'. Bildiğiniz gibi bir dil modelini (LLM) daha iyi hale getirmek veya spesifik bir göreve ince ayar (fine-tune) yapmak istediğimizde PPO, GRPO gibi karmaşık, ağır ve pahalı optimizasyon algoritmaları kullanırız. Bunlar, ağırlıkları yavaş yavaş, gradyan hesaplayarak günceller. 
Ama bu paper diyor ki: **Büyük modellerde (örneğin 3B veya 7B üstü) iyi çözümler aslında mevcut ağırlıkların hemen etrafında, çok yoğun bir 'çalılık' şeklinde bulunuyor!** Yani gradyan hesaplamadan, sadece ağırlıklara **rastgele gürültü** ekleyip, en iyi çalışanları seçip ensemble (çoğunluk oylaması) yapsak bile, geleneksel yöntemler kadar hatta onlardan daha iyi performans alabiliyoruz! Kulağa çılgınca geliyor değil mi? Maliyet sıfıra iniyor, iletişim maliyeti yok ve süreç tamamen paralel! Hadi gelin, bu inanılmaz **RandOpt** algoritmasını kendi bilgisayarımızda nasıl çalıştıracağımızı adım adım görelim."

---

## 💻 Bölüm 1: Mantığı Anlamak (1:30 - 3:00)
**Görsel:** Ekranda basit bir iki aşamalı şema çizimi veya paper'daki Figür 1.
**Konuşma:**
"RandOpt algoritması sadece iki basit adımdan oluşuyor:
1. **Eğitim (Paralel):** Elimizdeki modelin (örneğin Qwen 2.5 3B) ağırlıklarına rastgele Gaussian gürültüsü (noise) ekliyoruz. Bunu N defa yapıp N farklı küçük modelcik oluşturuyoruz. Sonra bu modelleri kendi küçük veri setimizde (diyelim ki matematik soruları) test ediyoruz. En yüksek puanı alan **K** adet modeli seçiyoruz. 
2. **Çıkarım (Ensemble):** Soru soracağımız zaman, seçilmiş bu K modelin hepsine aynı soruyu soruyoruz. Çıkan cevaplar arasında 'çoğunluk oylaması' (Majority Voting) yapıp nihai cevabı veriyoruz.
Küçük modellerde bu işe yaramıyor çünkü iyi çözümler samanlıkta iğne gibi nadir. Ama büyük modellerde, uzay o kadar geniş ki, iyi çözümler tıpkı bir çalılık gibi çok yoğun!"

---

## 🛠 Bölüm 2: Kurulum ve Veri Seti (3:00 - 5:30)
**Görsel:** Ekranda VS Code açık. Terminal üzerinden işlemleri gösteriyoruz.
**Konuşma:**
"Hemen koda geçelim. GitHub reposunu clonluyoruz. Biz bugün kolay göstermek adına küçük bir 'Matematik Veri Seti' (dummy_math) oluşturacağız. RandOpt reposu, kendi veri setimizi eklememiz için bize çok kolay 3 adımlık bir yapı sunmuş:
1. JSON formatında soru-cevap verilerimizi oluşturacağız.
2. Bu soruların doğruluğunu ölçecek bir 'Reward' (Ödül) fonksiyonu yazacağız.
3. Modelden dönen cevabı yakalayacak bir 'Data Handler' (Veri Yakalayıcı) tanımlayacağız.

Sizin için hazırladığım `prepare_randopt_demo.py` scripti tüm bu dosyaları otomatik olarak oluşturuyor ve model klasörüne entegre ediyor."
*(Videoda kodu çalıştırın)*
`python prepare_randopt_demo.py`

---

## 🚀 Bölüm 3: Modeli RandOpt İle Eğitmek (5:30 - 8:00)
**Görsel:** `run_demo.sh` scriptini gösterin ve terminalde çalıştırın.
**Konuşma:**
"İşte zurnanın zırt dediği yer burası. RandOpt'un ana dosyasını çalıştırıyoruz. Parametrelere dikkat edelim:
- `--dataset tutorial_math`: Kendi tanımladığımız veri seti.
- `--population_size 500`: Modele 500 farklı rastgele gürültü ekleyerek 500 farklı perturbasyon (varyasyon) yaratacağız.
- `--sigma_values "0.0005,0.001"`: Eklenen gürültünün şiddeti. Çok gürültü modeli bozar, azı işe yaramaz. Optimizasyon burada yapılıyor.
- `--top_k_ratios "0.04"`: Yani en iyi %4'lük kısmı, yani 500 varyasyondan en başarılı 20 modeli oylama için seçeceğiz.

Bu işlemi çalıştırdığımız anda, modelinizdeki GPU'lar tamamen otonom ve paralel şekilde bu ağırlıklara gürültü ekleyip, test veri setindeki performansı tartacak. Görüyorsunuz asıl devrim burada: Backpropagation yok! Gradyan hesabı yok! Sadece forward-pass yani tahmin yapıyoruz ve işe yarayanı ayırıyoruz."

---

## 📊 Bölüm 4: Sonuçlar ve Kapanış (8:00 - 10:00)
**Görsel:** Çalışma bitince ekrana gelen çıktı klasörünü, JSON loglarını (`experiment_dir`) gösterin.
**Konuşma:**
"İşlem tamamlandı! RandOpt bizim için gürültü eklenmiş en iyi ağırlık dosyalarının loglarını kaydetti. Paper'da da bahsedildiği gibi, bazı varyasyonlar matematik sorularında çok iyi çıkarken, bazıları kimyada iyi olabilir. Kötü olanları eledik, iyi olanları topluluğa kattık.

Özetle, 'Neural Thickets' ve RandOpt bize şunu söylüyor: **Yeterince büyük ve güçlü bir modeli baştan eğittiyseniz, ince ayar yapmak için zeki ve yorucu algoritmalarla cebelleşmeye gerek yok. Rastgele şans ve iyi bir eleme mekanizması işi çözebilir!**
Açıklamadaki GitHub linkinden bu videodaki materyallere ulaşabilirsiniz. RandOpt ile kendi veri setlerinizde neler yaptığınızı yorumlarda benimle paylaşın. Beğenmeyi ve abone olmayı unutmayın, bir sonraki devrim niteliğindeki makalede görüşmek üzere, hoşçakalın!"
