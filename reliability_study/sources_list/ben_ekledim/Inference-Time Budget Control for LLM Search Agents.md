# Makalenin ana fikri

Bu makale, araç kullanan LLM ajanlarının—örneğin web araması, belge getirimi veya soru parçalama yapan sistemlerin—**sınırlı araç çağrısı ve çıktı tokenı bütçesini** daha akıllıca harcamasını hedefliyor.

Temel soru şu:

> Bir arama ajanının elinde az sayıda arama hakkı ve sınırlı cevap üretme tokenı varken, sıradaki bütçeyi **arama yapmaya mı**, **soruyu alt sorulara bölmeye mi**, yoksa **artık cevap vermeye mi** harcaması gerekir?

Yazarların cevabı iki aşamalı bir kontrol mekanizması:

1. **Arama sırasında:** Her olası eylemin, kalan bütçe başına sağlayacağı tahmini faydayı hesaplayıp en değerli eylemi seçmek.
2. **Arama bittikten sonra:** Bulunan cevabı yalnızca düşük riskli, biçimsel bir hata varsa düzeltmek; aksi halde gereksiz “yeniden yazma” yapmayıp ilk cevabı korumak.

Makalenin iddiası, daha fazla arama ya da daha çok token kullanmanın otomatik olarak daha iyi yanıt anlamına gelmediği; asıl önemli olanın, **bütçenin hangi karar noktasına harcandığı** olduğudur. [Introduction](https://www.alphaxiv.org/abs/2605.05701?page=1)

---

# 1. Problem neden önemli?

Klasik LLM kullanımında model tek seferde cevap üretir. Ancak modern ajanlarda model:

- arama motoruna sorgu atar,
- gelen belgeleri okur,
- soruyu parçalara ayırır,
- farklı bilgi yollarını dener,
- sonra nihai yanıtı yazar.

Bu yaklaşım, ReAct ve Toolformer gibi çalışmalardan sonra yaygınlaştı. Fakat araç kullanımı ücretsiz değildir:

- Her arama çağrısının gecikme ve maliyeti vardır.
- Her üretilen ara düşünce veya özet token tüketir.
- Fazla arama, aynı bilgiyi tekrar getirebilir.
- Fazla dallanma, ajanı daha çok kanıta değil daha çok gürültüye götürebilir.
- Geç cevap vermek bütçeyi tüketir; çok erken cevap vermek ise kritik kanıtı kaçırır.

Makale bu yüzden problemi yalnızca “daha fazla test-time compute kullanma” meselesi olarak görmüyor. Bunun yerine, bunu **kısıtlı kaynak altında sıralı karar verme** problemi olarak formüle ediyor. Yani ajan her adımda şunu soruyor:

$$
\text{Sıradaki eylem, kalan bütçe başına en yüksek görev faydasını verir mi?}
$$

Yazarlar özellikle iki ayrı bütçeyi birlikte ele alıyor:

- $B_{\text{tool}}$: izin verilen araç/arama çağrısı sayısı,
- $B_{\text{tok}}$: izin verilen çıktı tokenı sayısı.

Bu “çift bütçe” önemli: Bir ajan araç çağrısı bakımından ekonomik olabilir ama çok uzun ara metinler üretip token bütçesini tüketebilir; ya da bunun tersi olabilir. [Problem Formulation](https://www.alphaxiv.org/abs/2605.05701?page=4)

---

# 2. Çalışmanın önerdiği çözüm: iki aşamalı kontrol

Makalenin sistemi, mevcut bir ağaç-arama omurgasının—BAVT esinli bir yapının—üzerine eklenen **eğitimsiz (training-free)** bir denetleyicidir.

Bu iki bileşeni ayırmak çok önemli:

| Aşama | Karar | Amaç |
|---|---|---|
| **Aşama 1: Arama-zamanı kontrolü** | Ara, parçala, yoksa cevap ver mi? | Bütçeyi arama süreci boyunca faydalı adımlara ayırmak |
| **Aşama 2: Cevap-zamanı sonlandırma** | İlk cevabı koru mu, düzeltilmiş adayı seç mi? | Kanıt yeterliyken küçük cevap-biçimi hatalarını güvenli biçimde düzeltmek |

Bu tasarımın ana sezgisi şudur:

- **Arama hatası** ile **cevap biçimi hatası** farklı problemlerdir.
- Eğer ajan yanlış belgeyi bulduysa veya çok-adımlı akıl yürütme zincirini kuramadıysa, son anda cevabı yeniden yazmak sorunu çözmez.
- Ama ajan doğru kanıtı topladıysa ve yalnızca “evet/hayır” kutbunu ters yazdıysa, tarihi eksik verdiyseniz ya da doğru varlığın yanlış takma adını kullandıysa; ihtiyatlı bir düzeltme yararlı olabilir.

Makale, bu ayrımın hem performans hem de güvenlik açısından gerekli olduğunu savunuyor. [Method](https://www.alphaxiv.org/abs/2605.05701?page=6)

---

# 3. Aşama 1 — Arama sırasında bütçe nasıl kontrol ediliyor?

## 3.1 Ajanın eylem uzayı

Her arama adımında ajan üç temel eylemden birini seçiyor:

1. **SEARCH:** Yeni kanıt/sonuç getirmek için arama yapmak.
2. **DECOMPOSE:** Soruyu alt sorulara ayırmak.
3. **ANSWER:** Yeterli kanıt olduğunu düşünüp cevap vermeye yönelmek.

Örneğin şu çok-hoplu soruyu düşünelim:

> “X şirketinin kurucusunun mezun olduğu üniversite hangi şehirde?”

Bunun için ajan:

- önce şirketin kurucusunu arayabilir,
- sonra kurucunun eğitim bilgisini arayabilir,
- sonra üniversitenin bulunduğu şehri bulabilir,
- veya elindeki kanıt yeterliyse doğrudan yanıt verebilir.

Kötü bir kontrolcü:

- daha fazla bilgiye ihtiyaç yokken tekrar tekrar arama yapabilir;
- zor bir çok-hop soruyu hiç parçalamadan cevaplayabilir;
- ilk belgeyi görünce aceleyle yanıt üretebilir.

Bu makalenin denetleyicisi ise her adımda bu üç seçeneği yeniden puanlıyor. [Challenges](https://www.alphaxiv.org/abs/2605.05701?page=2)

---

## 3.2 Task-level VOI nedir?

Makalenin merkezindeki kavram **task-level Value of Information (VOI)**, yani görev düzeyinde “bilginin değeri”dir.

Klasik bilgi teorisindeki Shannon bilgi kazancını hesaplamıyor. Bunun yerine daha operasyonel bir soru soruyor:

> “Bu eylemi şimdi yaparsam, mevcut durum ve kalan bütçe altında nihai cevap kalitesine marjinal katkısı ne olur; maliyetine değer mi?”

Her eylem $k$ için önce bir fayda hesaplanıyor:

$$
u_t(k)
=
\hat{\Delta}_t(k)
+
\Psi_t(k)
-
\Pi_t(k; b_t)
$$

Burada:

- $\hat{\Delta}_t(k)$: Eleştirici/critic tarafından tahmin edilen ilerleme; bu eylem cevabı ne kadar iyileştirebilir?
- $\Psi_t(k)$: Yapısal sinyaller; örneğin çok-hoplu “köprü” yapısı var mı, ajan döngüye mi girdi, cevap vermek için kanıt yeterli mi?
- $\Pi_t(k;b_t)$: Bütçe cezası; bütçe daraldıkça ek arama ve soru parçalama daha pahalı görünür.

Sonra bu ham yarar, eylemin bütçe maliyetine bölünüyor:

$$
r_t(k)
=
\frac{[u_t(k)]_+}{d_t(k;b_t)+\epsilon}
$$

Buradaki $r_t(k)$, makalenin **VOI puanı**dır: “birim bütçe başına pozitif görev değeri” tahmini. Sistem en yüksek VOI puanlı eylemi seçer. [VOI Score](https://www.alphaxiv.org/abs/2605.05701?page=6)

### Neden maliyete bölmek gerekiyor?

Şöyle düşünün:

- Bir arama eylemi teorik olarak biraz daha faydalı olabilir,
- fakat iki araç çağrısı ve uzun bir ara açıklama gerektirebilir.
- Buna karşılık cevap vermek daha az faydalı görünse de neredeyse ücretsiz olabilir.

Ham faydayı maksimize etmek, pahalı eylemleri gereğinden fazla seçmeye yol açar. VOI ise yaklaşık olarak şunu maksimize eder:

$$
\frac{\text{Beklenen cevap-kalitesi artışı}}{\text{Harcanacak bütçe}}
$$

Bu nedenle öneri, “en faydalı eylemi” değil, **maliyetine göre en değerli eylemi** seçer.

---

## 3.3 Bütçe baskısı

Yazarlar kalan bütçeyi şu tür bir baskı sinyaliyle temsil ediyor:

$$
\rho_t
=
1-\min\left\{
\frac{b_{\text{tool},t}}{B_{\text{tool}}},
\frac{b_{\text{tok},t}}{B_{\text{tok}}}
\right\}
$$

Burada $\rho_t$ büyüdükçe, iki bütçeden en az biri kritik biçimde azalıyor demektir.

Bu durumda sistem:

- yeni **SEARCH** eylemlerini daha fazla cezalandırır,
- **DECOMPOSE** eylemlerine daha ihtiyatlı yaklaşır,
- eğer yeterli destek varsa **ANSWER** eylemini görece daha çekici hale getirir.

Ancak bu “bütçe azalınca hemen cevapla” kuralı değildir. Çünkü denetleyicide koruyucu kurallar da vardır. Örneğin kanıt zayıfsa veya soru bileşimsel görünüyorsa erken cevap verme bastırılabilir. [Implementation](https://www.alphaxiv.org/abs/2605.05701?page=32)

---

## 3.4 Koruyucu kurallar neden gerekli?

Sadece VOI puanı kullanmak bazı riskler doğurabilir. Örneğin bütçe azaldığında cevap verme eylemi ucuz olduğu için aşırı avantajlı görünebilir. Bu yüzden sistemin üstünde deterministik “guard” kuralları var.

Bu kurallar:

- Zayıf kanıt varken erken cevap vermeyi engeller.
- Basit, tek-faktörlü sorularda gereksiz parçalamayı engeller.
- Aynı şekilde tekrar eden ve ilerleme yaratmayan decomposition adımlarını azaltır.
- Yeterince bileşimsel sorularda en az bir arama yapılmasını zorunlu tutar.

Böylece sistem yalnızca maliyet odaklı davranmaz; sorunun yapısını da hesaba katar. Yazarlar bu kuralların arama altyapısını, retrieval backend’ini veya bütçe muhasebesini değiştirmediğini; yalnızca eylem puanını etkilediğini belirtiyor. [Fixed Coefficients](https://www.alphaxiv.org/abs/2605.05701?page=33)

---

# 4. Aşama 2 — Nihai cevap neden ve nasıl düzeltiliyor?

Arama tamamlandıktan sonra sistem iki aday cevaba sahip:

- $a_{\text{base}}$: Arama yörüngesinin ürettiği ana cevap.
- $a_{\text{ref}}$: Aynı arama izi ve aynı toplanmış kanıtlar üzerinden oluşturulmuş düzeltilmiş aday.

Önemli nokta: Bu aşama **yeni arama yapmıyor**, **ek araç çağrısı eklemiyor** ve **yeni LLM çağrısı yapmıyor**. Yani amaç daha fazla düşünmek değil; eldeki kanıtla iki aday arasında kontrollü seçim yapmak. [Finalization](https://www.alphaxiv.org/abs/2605.05701?page=7)

## 4.1 Her cevap neden yeniden yazılmıyor?

Çünkü yeniden yazma zararlı olabilir.

Çok-hoplu sorularda cevap genellikle bir “köprü varlık” üzerinden kurulur. Örneğin:

1. kişi $\rightarrow$ kurum,
2. kurum $\rightarrow$ tarih,
3. tarih $\rightarrow$ istenen sonuç.

Akıcı görünen bir yeniden yazma:

- köprü varlığı silebilir,
- karşılaştırma yönünü ters çevirebilir,
- kısa ama daha az doğru bir cevap oluşturabilir,
- doğru özgül cevabı daha genel ama yanlış bir ifadeyle değiştirebilir.

Bu yüzden makale, genel amaçlı “cevap editörü” yerine **seçici müdahale** tasarlıyor. [Answer Errors](https://www.alphaxiv.org/abs/2605.05701?page=2)

---

## 4.2 Güvenli düzeltme türleri

Finalizer sadece düşük riskli cevap-biçimi hatalarında müdahale etmeyi hedefliyor:

- **Evet/hayır kutbu hatası:** “Yes” yerine “No”.
- **İkili seçenek hatası:** A/B arasında yanlış seçimi düzeltmek.
- **Türlenmiş slot hatası:** yıl, tarih, kapasite, sayı gibi alanlardaki biçimsel yanlışlıklar.
- **Desteklenen factoid tamamlama:** kanıt açıkça cevabı destekliyorsa eksik kısa yanıtı tamamlamak.

Buna karşılık şu durumlarda sistem müdahaleden kaçınıyor:

- Köprü akıl yürütme hâlâ çözülmemişse,
- Karşılaştırmalı anlam belirsizse,
- Doğrudan kanıt eksikse.

Bu, sistemin güçlü tarafının “yanıt formatı doğruluğu” olduğunu; yanlış retrieval yolunu veya bozuk çok-adımlı muhakemeyi onaramadığını açıkça ortaya koyuyor. [Final Answer Control](https://www.alphaxiv.org/abs/2605.05701?page=9)

---

## 4.3 Teorik karar kuralı

Makale teorik olarak şu tür bir seçim kuralı türetiyor:

$$
F(z)=G(z)-\eta H(z)
$$

- $z$: yörünge ve iki aday cevaptan çıkarılan özellikler,
- $G(z)$: düzeltmenin beklenen getirisi,
- $H(z)$: düzeltmenin zarar verme riski,
- $\eta$: riskten kaçınma katsayısı.

Karar:

$$
\hat{a}
=
\begin{cases}
a_{\text{ref}}, & z \in S_{\text{safe}} \text{ ve } F(z)\ge \tau \\
a_{\text{base}}, & \text{aksi halde}
\end{cases}
$$

Yani düzeltilmiş cevap ancak:

1. vaka güvenli kümeye giriyorsa,
2. beklenen faydası müdahale riskini aşıyorsa

kabul ediliyor. [Finalization Rule](https://www.alphaxiv.org/abs/2605.05701?page=6)

Fakat burada kritik bir teorik sınırlama var: Teoremde kullanılan gerçek koşullu kazanç ve zarar miktarları ($G^\star$, $H^\star$) pratikte bilinmez. Bu nedenle yayımlanan sistem teoremi doğrudan uygulamıyor; bunun yerine açık özellik koşullarına dayanan muhafazakâr bir deterministik kural kullanıyor. Yazarlar bunu açıkça kabul ediyor ve teorik üst sınır ile pratik kural arasındaki performans farkını açık problem olarak bırakıyor. [Released Finalizer](https://www.alphaxiv.org/abs/2605.05701?page=30)

---

# 5. Deney düzeni

Yazarlar sistemi dört çok-hoplu soru-cevap veri setinde değerlendiriyor:

- HotpotQA
- 2WikiMultihopQA
- MuSiQue
- Bamboogle

Üç farklı LLM omurgası kullanılıyor:

- [Qwen3](https://www.alphaxiv.org/abs/2505.09388)-32B,
- Qwen3.5-122B,
- GPT-5.4-Mini.

Karşılaştırılan yöntemler:

- BAVT,
- BATS,
- AFlow,
- [Search-o1](https://www.alphaxiv.org/abs/2501.05366),
- önerilen VOI denetleyicisi.

Dört bütçe seviyesi var:

| Seviye | Araç çağrısı üst sınırı | Çıktı token üst sınırı |
|---|---:|---:|
| Düşük | 1 | 100 |
| Düşük-orta | 2 | 200 |
| Yüksek-orta | 2 | 300 |
| Yüksek | 3 | 500 |

Değerlendirme “sert” biçimde yapılıyor: Bir örnek araç çağrısı ya da token sınırlarından herhangi birini aşarsa başarısız sayılıyor. Bu, yöntemlerin rahat bırakılmış maliyetlerle değil, **aynı sıkı bütçe altında** karşılaştırıldığı anlamına gelir. [Experimental Setup](https://www.alphaxiv.org/abs/2605.05701?page=7)

---

# 6. Ana deney sonuçları

## 6.1 En büyük fayda düşük ve orta bütçelerde

Qwen3-32B üzerinde VOI yöntemi:

- 16 veri seti-bütçe hücresinin **7’sinde en iyi F1** sonucunu alıyor,
- **2 ek hücrede** en iyiyle berabere kalıyor,
- BAVT’yi 16 hücrenin 15’inde,
- Search-o1’i 16 hücrenin tamamında geride bırakıyor.

Yazarların yorumuna göre en büyük iyileşmeler düşük ve düşük-orta bütçe seviyelerinde görülüyor. Bu mantıklı: Bütçe çok kısıtlıyken tek bir gereksiz retrieval veya erken cevap kararı bütün yörüngeyi bozabilir. [Main Results](https://www.alphaxiv.org/abs/2605.05701?page=8)

Ancak makale “her yerde en iyi” iddiasında değil:

- BATS bazı yüksek bütçe koşullarında güçlü kalıyor.
- AFlow, 2WikiMultihopQA’nın düşük-orta bütçesinde lider oluyor.
- Qwen3.5-122B üzerinde sonuçlar daha karışık; yüksek bütçelerde BATS ve BAVT bazı hücrelerde öne geçiyor.

Bu önemli bir bulgu: Denetleyicinin kazancı, temel model güçlendikçe veya bütçe genişledikçe azalabiliyor. Yani VOI, her koşulda üstün bir arama algoritması olmaktan çok, **bütçe kıt olduğunda eylem tahsisinin değerini artıran bir katman**. [Limitations](https://www.alphaxiv.org/abs/2605.05701?page=10)

---

## 6.2 Ablasyon: hangi bileşen gerçekten önemli?

Yazarlar Qwen3-32B ve yüksek-orta bütçede denetleyicinin parçalarını çıkararak test ediyorlar.

| Veri seti | BAVT F1 | VOI tam F1 | Kazanç |
|---|---:|---:|---:|
| HotpotQA | 0.41 | 0.47 | +0.06 |
| 2WikiMultihopQA | 0.45 | 0.63 | +0.18 |
| MuSiQue | 0.34 | 0.43 | +0.09 |
| Bamboogle | 0.42 | 0.56 | +0.14 |

<!-- Bar Chart: Qwen3-32B, yüksek-orta bütçe: BAVT ve tam VOI F1 -->

Bu tabloda en çarpıcı sonuç, **bütçe-bağımlı cezanın çıkarılması**. Bu bileşen kaldırıldığında her veri setinde performans düşüyor; örneğin 2WikiMultihopQA’da F1, tam VOI’nin $0.63$ değerinden $0.43$’e iniyor. Bu yüzden makalenin ana teknik sonucu yalnızca “bir VOI skoru kullandık” değil; özellikle kalan bütçe durumunu eylem seçiminde açık biçimde hesaba katmanın kritik olduğu. [Ablation Results](https://www.alphaxiv.org/abs/2605.05701?page=9)

---

## 6.3 Gecikme sonucu

Denetleyici ek bir hesaplama katmanı olmasına rağmen, Qwen3-32B ve yüksek-orta bütçe testinde BAVT’ye göre ortalama çalışma süresini:

$$
20.91\text{s} \rightarrow 15.23\text{s}
$$

seviyesine indiriyor; bu **%27.2 daha düşük ortalama duvar-saati süresi** demek.

Bu sonuç ilk bakışta şaşırtıcı olabilir: Denetleyici ekleniyor ama sistem hızlanıyor. Açıklaması, denetleyicinin kendi hesaplama maliyetinin küçük olması; buna karşılık gereksiz retrieval, tekrar eden decomposition ve düşük değerli yörünge adımlarını azaltması. Yine de 2WikiMultihopQA’da bazı ablation varyantları tam sistemden daha hızlı; dolayısıyla hız üstünlüğü her veri kümesinde aynı mekanizmayla ortaya çıkmıyor. [Inference Cost](https://www.alphaxiv.org/abs/2605.05701?page=9)

---

## 6.4 İkinci aşama ne kadar katkı sağlıyor?

Makalenin sonuçlarına göre toplam iyileşmenin çoğu Aşama 1’den, yani arama-zamanı bütçe tahsisinden geliyor.

BAVT’ye karşı tam sistemin göreli F1 artışları:

- HotpotQA: **%5.7**
- 2WikiMultihopQA: **%11.8**
- MuSiQue: **%14.7**
- Bamboogle: **%18.4**

Aşama 2’nin katkısı:

- HotpotQA’da ek kazanç yok,
- 2WikiMultihopQA’da toplam kazancın %13.4’ü,
- MuSiQue’de %41.9’u,
- Bamboogle’da %27.8’i.

Bu desen, finalizer’ın arama kalitesini değil, belirli cevap-biçimi kusurlarını düzelttiğini doğruluyor. Arama yolu zaten yeterince iyi olduğunda yararlı; ama temel problem eksik köprü akıl yürütmeyse etkisi sınırlı. [Two-Stage Ablation](https://www.alphaxiv.org/abs/2605.05701?page=10)

---

# 7. Teorik kısım ne söylüyor?

Makale iki ana teorik iddia sunuyor.

## 7.1 Arama-zamanı teorisi

İlk teorem, kullanılan eylem fayda skorunun ideal fakat erişilemeyen bir “oracle” tek-adım ileri bakış değerini yerel olarak yaklaşık hesapladığını söylüyor.

Yani eğer:

- critic’in ilerleme tahmini makulse,
- yapısal sinyaller makulse,
- bütçe cezası gerçek fırsat maliyetini iyi yakalıyorsa,
- eylemler arasında yeterince belirgin fark varsa,

VOI sıralaması oracle’ın eylem sıralamasıyla uyumlu olabilir.

Fakat bu bir **küresel optimalite** garantisi değildir. Makale, tam arama ağacının en iyi çözümünü bulduğunu iddia etmiyor. Sadece her bir anlık kararın, ideal bütçe-duyarlı kararın iyi bir yerel yaklaşımı olabileceğini savunuyor. [Theoretical Support](https://www.alphaxiv.org/abs/2605.05701?page=7)

## 7.2 Cevap-zamanı teorisi

İkinci teorem, zarar-kısıtlı cevap değiştirme probleminde en iyi politikanın bir eşik kuralı biçiminde olduğunu gösteriyor:

- Beklenen fayda,
- risk katsayısı ile ağırlıklandırılmış beklenen zarardan yüksekse,
- ve vaka güvenli kümedeyse

düzeltilmiş cevabı seç.

Bu, pratikte kullanılan kuralın teorik ilhamı. Ancak yukarıda değindiğim gibi gerçek $G^\star(z)$ ve $H^\star(z)$ değerleri bilinmediği için yayımlanan sistem bunları doğrudan hesaplayamıyor. [Answer Replacement Theorem](https://www.alphaxiv.org/abs/2605.05701?page=26)

---

# 8. Makalenin güçlü yanları

## 8.1 Gerçekçi maliyet modeli

Çoğu test-time scaling çalışması token sayısına odaklanır. Burada araç çağrıları ve tokenlar birlikte ele alınıyor. Gerçek ajan sistemlerinde bu daha gerçekçi; web arama, API çağrısı, retrieval gecikmesi ve üretilen bağlam uzunluğu farklı maliyet kaynaklarıdır.

## 8.2 Aynı sert bütçe altında karşılaştırma

Sistemlerin aynı araç ve token kısıtları altında değerlendirilmesi güçlü bir deneysel tercih. Böylece bir yöntemin daha iyi görünmesinin nedeni “daha fazla kaynak tüketmesi” olmuyor.

## 8.3 “Daha çok arama her zaman daha iyi değildir” bulgusu

Makalenin en değerli mesajlarından biri bu. Yüksek bütçeler bazen daha kötü sonuç veriyor; çünkü ek arama tekrara, bağlam şişmesine veya yanlış yönlendiren kanıta yol açabiliyor. Bu, [THOUGHTTERMINATOR](https://www.alphaxiv.org/abs/2504.13367) gibi “overthinking” çalışmalarının ajanik arama bağlamındaki karşılığı olarak düşünülebilir. [Discussion](https://www.alphaxiv.org/abs/2605.05701?page=10)

## 8.4 Finalizer’ın muhafazakâr oluşu

Birçok sistem “yeniden düşün / yeniden yaz” yaklaşımını varsayılan olarak kullanır. Bu makale ise doğruluk açısından bazen en iyi eylemin **hiç müdahale etmemek** olduğunu net biçimde modeller. Çok-hoplu QA için bu doğru bir tasarım sezgisidir.

---

# 9. Sınırlamalar ve eleştiriler

## 9.1 VOI gerçek bir öğrenilmiş değer fonksiyonu değil

“Value of Information” adı güçlü görünse de uygulamada bu skor:

- sabit katsayılar,
- açık özellikler,
- el yapımı cezalar,
- deterministik guard kuralları

üzerine kurulu.

Bu nedenle yöntem, tam anlamıyla veriden öğrenilmiş veya kalibre edilmiş bir karar teorisi çözümü değil. “VOI” burada sıkı bir Bayesçi VOI hesabından ziyade, görev-fayda/bütçe oranını temsil eden bir tasarım çerçevesi.

## 9.2 Katsayıların genellenebilirliği açık soru

Makale, katsayıların veri seti ya da omurga başına ayarlanmadığını söylüyor. Bu iyi bir genellenebilirlik iddiası. Fakat maliyet-ceza ölçeği, decomposition bonusu ve erken-cevap cezası gibi değerlerin neden bu sayılar olduğu daha ayrıntılı biçimde gerekçelendirilebilirdi. Farklı araç maliyetleri, farklı retrieval kalitesi veya gerçek web ortamları altında aynı sabitler iyi çalışmayabilir. [Controller Settings](https://www.alphaxiv.org/abs/2605.05701?page=33)

## 9.3 Değerlendirme alanı sınırlı

Deneyler çok-hoplu QA üzerinde yapılıyor. Bu, yöntem için uygun bir test alanı; fakat doğrudan şunlara genellenmiş değil:

- açık uçlu web araştırması,
- kod ajanları,
- bilgisayar kullanımı,
- uzun-horizon görevler,
- araç çıktılarının güvenilmez ya da çelişkili olduğu ortamlar.

Dolayısıyla çalışma, genel amaçlı ajan bütçe kontrolünü tamamen çözmüyor; daha dar biçimde, **retrieval-ağırlıklı çok-hoplu QA ajanlarında** eylem tahsisini iyileştiriyor.

## 9.4 Yüksek bütçede üstünlük tutarlı değil

BATS ve BAVT’nin bazı yüksek-bütçe hücrelerinde daha iyi olması, VOI yaklaşımının keşif alanı genişlediğinde her zaman en iyi strateji olmayabileceğini gösteriyor. Daha geniş bütçede beam-style kanıt toplama veya daha agresif dallanma yararlı hale gelebilir.

Bu da yöntem için iyi bir sınır tanımı veriyor:

> VOI özellikle kıt kaynak koşullarında güçlü; bol kaynakta ise arama genişliği ve model kapasitesi daha baskın hale geliyor.

---

# 10. Sonuç: Bu makaleyi tek cümlede nasıl özetleriz?

Bu çalışma, araç kullanan LLM ajanlarında “daha fazla düşünme/arama” yerine **doğru anda doğru eyleme bütçe ayırma** fikrini savunuyor: arama sırasında retrieval, soru parçalama ve cevap verme arasında maliyet-fayda temelli seçim yapıyor; arama sonrasında ise yalnızca düşük riskli cevap-biçimi hatalarını düzeltip gereksiz yeniden yazmadan kaçınıyor.

En güçlü deneysel mesajı:

- Kazancın büyük bölümü **arama-zamanı bütçe kontrolünden** geliyor.
- Özellikle bütçenin çok sınırlı olduğu koşullarda faydalı.
- Nihai cevap düzeltmesi yalnızca kanıt doğru ama cevap biçimi hatalı olduğunda işe yarıyor.
- Yöntem bütün veri setlerinde, bütçelerde ve omurgalarda mutlak üstün değil; yüksek bütçelerde ve daha güçlü modellerde avantajı azalabiliyor. [Conclusion](https://www.alphaxiv.org/abs/2605.05701?page=11)