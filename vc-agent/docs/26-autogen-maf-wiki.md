# AutoGen ve MAF — çerçeve wiki'si

> **Bu ne:** AutoGen'i kullanacak ya da MAF'a geçmeyi düşünen bir mühendis için.
> Tek dosya, arayarak okunmak için.
>
> **Sürümler:** `autogen-core` / `agentchat` / `ext` **0.7.5** · `agent-framework`
> **1.14.0** — ikisi de kurulu ve ölçüldü.
>
> **Etiketler:** **[ölçüldü]** koşturuldu · **[kaynak]** birincil metinden ·
> **[teyitsiz]** okundu, koşturulmadı.

---

## İçindekiler

1. [Dört isim, tek karmaşa](#s1)
2. [Üç katman](#s2)
3. [Aktör modeli](#s3)
4. [Kimlik: bir şey değil, iki şey](#s4)
5. [İki iletişim biçimi](#s5)
6. [Tool döngüsü ve tarif](#s6)
7. [Beş takım tipi ve faturaları](#s7)
8. [Durmayı öğretmek](#s8)
9. [Sekiz resmî desen](#s9)
10. [Built-in tool'lar — ve neden yok](#s10)
11. [Kod yürütücüler](#s11)
12. [Ölçülmüş tuzaklar](#s12)
13. [MAF: halef ne getirdi](#s13)
14. [MAF: ne kaybettirdi](#s14)
15. [Geçiş haritası](#s15)

---

<a id="s1"></a>
## 1 · Dört isim

Karıştırılan dört ayrı şey var:

| İsim | Ne | Durum |
|---|---|---|
| **microsoft/autogen v0.4+** | `autogen-core` + `agentchat` + `ext` | Bakım modu, **0.7.5** |
| AutoGen v0.2 | `ConversableAgent`, `initiate_chat` | Terk edilmiş |
| **ag2ai/ag2** | v0.2 kolundan fork, ayrı ekip | Aktif — ama `pip install ag2` artık `import autogen` **sunmuyor** |
| **microsoft/agent-framework** | AutoGen + Semantic Kernel birleşimi | Resmî halef, **1.14.0** |

> **Filtre:** Bir kaynakta `ConversableAgent` ya da `initiate_chat` görüyorsan
> o kaynak v0.2 ya da AG2 anlatıyor — bu sürümle **uyumsuz**.

---

<a id="s2"></a>
## 2 · Üç katman

<div align="center">
<svg viewBox="0 0 600 182" width="600" height="182"><path d="M21,13 L579,13 L579,47 L21,47 Z" fill="#f8f9fa" stroke="none"/><path d="M19.9,12.2 Q300.0,12.0 579.6,13.1 M579.8,12.8 Q580.5,30.0 580.3,49.3 M579.5,47.4 Q300.0,48.8 18.8,46.6 M19.4,48.3 Q20.0,30.0 21.5,13.6" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><path d="M19.5,11.6 Q300.0,13.0 581.3,11.1 M579.2,11.8 Q579.5,30.0 580.2,49.5 M581.1,47.5 Q300.0,47.2 19.5,49.2 M20.7,46.7 Q20.1,30.0 18.6,11.8" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><text x="34" y="28" font-size="9.4" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">autogen_ext</text><text x="34" y="42" font-size="7.6" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">dış dünya</text><text x="210" y="36" font-size="8" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">model istemcileri · MCP · kod yürütücüler</text><path d="M21,57 L579,57 L579,91 L21,91 Z" fill="#e7f5ff" stroke="none"/><path d="M19.3,57.0 Q300.0,55.3 581.3,55.1 M579.0,55.0 Q580.6,74.0 580.7,91.6 M578.5,92.1 Q300.0,90.3 18.8,93.1 M19.2,92.3 Q19.8,74.0 21.2,54.4" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M21.3,55.7 Q300.0,55.0 578.7,57.4 M579.8,55.0 Q579.4,74.0 578.5,92.8 M580.8,92.6 Q300.0,91.0 19.5,91.3 M20.9,91.9 Q20.1,74.0 20.0,55.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="34" y="72" font-size="9.4" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">autogen_agentchat</text><text x="34" y="86" font-size="7.6" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">günlük iş</text><text x="210" y="80" font-size="8" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">AssistantAgent · 5 takım · 11 sonlandırma</text><path d="M21,101 L579,101 L579,135 L21,135 Z" fill="#f8f0fc" stroke="none"/><path d="M21.5,99.3 Q300.0,101.2 581.5,101.0 M579.2,100.6 Q580.1,118.0 580.1,136.3 M581.5,136.7 Q300.0,134.2 20.1,134.9 M21.0,135.2 Q19.4,118.0 20.6,99.4" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M20.6,98.9 Q300.0,99.0 578.6,101.6 M579.6,100.9 Q580.1,118.0 579.1,136.9 M580.7,136.8 Q300.0,137.0 18.7,135.6 M20.0,135.8 Q20.5,118.0 18.6,98.9" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="34" y="116" font-size="9.4" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">autogen_core</text><text x="34" y="130" font-size="7.6" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">aktör modeli</text><text x="210" y="124" font-size="8" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">AgentId(type,key) · runtime · topic · abonelik</text><text x="20" y="156" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Ayıran şey en alt katman: ajanlar gerçekten aktör — kendi mailbox'ı olan, mesajı tipe göre yönlendiren birimler.</text><text x="20" y="170" font-size="7.6" fill="#2f9e44" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Yukarıdan başla; aşağı inmek her zaman mümkün.</text></svg>
</div>

<sub>▲ AutoGen'in üç katmanı · düzenlemek için: [`f_layers.excalidraw`](diagrams/wiki/f_layers.excalidraw) → excalidraw.com'a sürükle</sub>


Ayıran şey **`autogen_core`**: ajanlar gerçekten aktör — kendi mailbox'ı olan,
mesajı **tipe göre** yönlendiren, makinelere dağıtılabilen birimler.

LangGraph'ın graf yürütücü + checkpointer'ı **dayanıklılık** sağlıyor,
eşzamanlılık modeli değil. *"AutoGen mı LangGraph mı"* çoğu zaman yanlış
sorulmuş soru — farklı katmanlar.

---

<a id="s3"></a>
## 3 · Aktör modeli

<div align="center">
<svg viewBox="0 0 600 180" width="600" height="180"><path d="M5.0,4.6 Q300.0,7.4 594.2,7.6 M593.5,5.3 Q594.1,90.0 593.3,173.3 M592.6,173.8 Q300.0,175.5 6.0,173.1 M6.1,173.3 Q7.3,90.0 5.8,5.3" fill="none" stroke="#868e96" stroke-width="1.2" stroke-linecap="round" stroke-dasharray="7 5"/><path d="M7.5,4.8 Q300.0,5.1 594.7,6.6 M593.6,7.1 Q594.8,90.0 592.9,173.0 M595.2,173.7 Q300.0,175.7 4.9,174.2 M6.9,172.7 Q5.8,90.0 6.3,7.3" fill="none" stroke="#868e96" stroke-width="1.2" stroke-linecap="round" stroke-dasharray="7 5"/><text x="13" y="19" font-size="7.2" fill="#868e96" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">SingleThreadedAgentRuntime</text><path d="M31,47 L147,47 L147,99 L31,99 Z" fill="#ffffff" stroke="none"/><path d="M30.2,47.4 Q89.0,45.8 147.9,46.0 M147.0,46.0 Q147.8,73.0 148.4,100.9 M147.4,98.7 Q89.0,101.5 31.0,100.6 M31.5,101.5 Q29.0,73.0 30.5,46.4" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><path d="M28.4,46.1 Q89.0,44.8 146.6,45.0 M146.5,45.9 Q148.6,73.0 147.8,101.1 M148.4,100.0 Q89.0,99.9 30.5,99.9 M31.6,101.6 Q29.5,73.0 31.1,46.7" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><text x="89.0" y="72.2" font-size="9.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Agent A</text><text x="89.0" y="83.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">type + key</text><path d="M241,35 L357,35 L357,73 L241,73 Z" fill="#e7f5ff" stroke="none"/><path d="M239.1,33.3 Q299.0,33.3 356.6,34.9 M359.1,33.6 Q358.2,54.0 359.5,75.1 M357.1,75.3 Q299.0,75.8 239.9,75.5 M238.6,74.4 Q239.8,54.0 240.9,33.3" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M239.5,35.5 Q299.0,32.5 358.8,32.8 M356.7,32.6 Q358.4,54.0 359.0,73.0 M357.8,73.0 Q299.0,73.8 240.7,72.8 M238.8,73.7 Q240.2,54.0 239.1,33.3" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="299.0" y="57.2" font-size="9.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Agent B</text><path d="M241,97 L357,97 L357,135 L241,135 Z" fill="#e7f5ff" stroke="none"/><path d="M241.0,95.4 Q299.0,97.7 359.2,95.1 M359.1,96.5 Q358.2,116.0 356.7,137.6 M357.2,136.9 Q299.0,137.0 239.5,135.3 M238.7,136.3 Q239.3,116.0 239.2,96.3" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M239.9,97.5 Q299.0,95.5 357.9,96.2 M357.0,94.9 Q357.4,116.0 359.3,137.0 M357.0,136.8 Q299.0,136.9 241.4,135.0 M241.2,136.3 Q240.7,116.0 239.7,94.7" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="299.0" y="119.2" font-size="9.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Agent C</text><path d="M453,47 L569,47 L569,99 L453,99 Z" fill="#ebfbee" stroke="none"/><path d="M453.5,45.2 Q511.0,44.3 570.7,45.2 M570.3,45.3 Q569.3,73.0 569.0,100.7 M569.1,100.2 Q511.0,101.6 453.1,100.4 M453.3,99.1 Q451.5,73.0 450.5,45.3" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M450.6,45.0 Q511.0,45.8 569.6,46.2 M569.6,47.3 Q570.8,73.0 571.5,100.5 M570.3,98.8 Q511.0,99.3 450.5,98.5 M452.6,101.5 Q452.9,73.0 450.5,46.4" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="511.0" y="72.2" font-size="9.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">ClosureAgent</text><text x="511.0" y="83.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">toplayıcı</text><path d="M150.7,65.4 Q194.0,59.9 239.6,52.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M150.8,67.3 Q194.0,60.2 238.8,54.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M238.0,54.0 L232.8,59.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M238.0,54.0 L231.1,51.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="194.0" y="54.0" font-size="7.4" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" text-anchor="middle">publish</text><path d="M151.2,81.7 Q193.4,98.7 238.9,113.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M150.4,81.6 Q193.9,97.3 238.3,112.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M238.0,112.0 L230.2,114.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M238.0,112.0 L234.1,106.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M359.6,54.8 Q404.9,61.0 450.3,65.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M358.7,54.8 Q404.8,61.5 448.5,66.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M450.0,66.0 L443.2,68.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M450.0,66.0 L444.7,61.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M361.3,115.2 Q405.2,99.5 448.4,81.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M359.0,114.9 Q405.3,99.7 451.3,82.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M450.0,82.0 L445.6,88.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M450.0,82.0 L442.5,80.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="30" y="122" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">mesaj kuyruğu · tek iş parçacığı · sıra korunur</text><text x="30" y="136" font-size="7.6" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">ajanlar birbirini çağırmaz — runtime taşır</text></svg>
</div>

<sub>▲ Ajan ajanı çağırmıyor · düzenlemek için: [`f_actor.excalidraw`](diagrams/wiki/f_actor.excalidraw) → excalidraw.com'a sürükle</sub>


Bir ajan başka bir ajanın nesnesini tutmuyor; runtime'a mesaj veriyor.

**Bedeli:** *"kim kimi çağırdı"* yığın izinde görünmüyor.
**Karşılığı:** yeni ajan eklemek çağıran kodu değiştirmiyor · bütün mesajlar tek
noktadan geçtiği için müdahale ve ölçüm oraya takılıyor · aynı sınıftan istediğin
kadar örnek bedava.

---

<a id="s4"></a>
## 4 · Kimlik

<div align="center">
<svg viewBox="0 0 600 172" width="600" height="172"><path d="M25,25 L273,25 L273,105 L25,105 Z" fill="#f8f0fc" stroke="none"/><path d="M24.5,24.5 Q149.0,23.9 272.9,22.4 M273.3,25.0 Q274.4,65.0 274.6,106.3 M274.5,104.9 Q149.0,105.8 23.8,104.9 M22.6,107.0 Q25.3,65.0 22.6,24.6" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M23.7,25.1 Q149.0,23.4 272.5,22.6 M274.0,22.7 Q272.6,65.0 275.6,107.4 M273.8,104.8 Q149.0,107.4 23.4,106.4 M24.6,104.6 Q22.9,65.0 22.9,25.0" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="149.0" y="39" font-size="9.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">AgentId</text><text x="46" y="62" font-size="9.6" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">type = "analyst"</text><text x="46" y="86" font-size="9.6" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">key  = "arxiv"</text><text x="24" y="122" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">type = DAVRANIŞ (hangi sınıf)</text><text x="24" y="136" font-size="7.6" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">key  = ÖRNEK (hangi kopya)</text><path d="M331,23 L439,23 L439,55 L331,55 Z" fill="#e7f5ff" stroke="none"/><path d="M329.7,22.3 Q385.0,21.6 439.9,21.6 M440.7,23.5 Q440.6,39.0 441.5,56.5 M439.6,56.6 Q385.0,56.5 330.6,54.8 M329.6,56.0 Q329.6,39.0 331.0,22.0" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M331.0,21.8 Q385.0,20.3 440.1,21.1 M439.6,22.9 Q440.1,39.0 438.7,56.2 M440.6,54.5 Q385.0,57.2 329.3,55.5 M328.6,55.9 Q330.2,39.0 329.1,21.3" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="385.0" y="42.2" font-size="7.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">analyst/arxiv</text><path d="M331,65 L439,65 L439,97 L331,97 Z" fill="#e7f5ff" stroke="none"/><path d="M331.5,63.9 Q385.0,63.9 441.5,62.9 M439.6,63.0 Q439.4,81.0 440.8,99.3 M439.3,98.9 Q385.0,99.8 329.5,97.2 M328.5,98.4 Q329.3,81.0 331.5,65.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M328.5,63.4 Q385.0,64.0 440.9,65.1 M439.1,64.1 Q439.8,81.0 439.9,98.4 M441.4,99.3 Q385.0,98.9 328.9,97.3 M329.3,99.4 Q330.5,81.0 330.0,64.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="385.0" y="84.2" font-size="7.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">analyst/hn</text><path d="M331,107 L439,107 L439,139 L331,139 Z" fill="#e7f5ff" stroke="none"/><path d="M329.1,105.3 Q385.0,106.7 439.2,106.6 M438.8,107.2 Q439.7,123.0 439.7,139.2 M441.5,140.2 Q385.0,140.2 330.4,141.3 M330.8,139.7 Q330.6,123.0 330.3,107.3" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M330.3,105.5 Q385.0,107.0 439.6,106.8 M438.5,106.7 Q439.9,123.0 441.3,141.0 M441.3,140.1 Q385.0,138.6 330.9,139.6 M329.5,139.4 Q329.5,123.0 331.4,106.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="385.0" y="126.2" font-size="7.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">analyst/gh</text><text x="456" y="46" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">aynı sınıf,</text><text x="456" y="60" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">üç ayrı örnek,</text><text x="456" y="74" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">üç ayrı durum</text><text x="330" y="158" font-size="7.6" fill="#868e96" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">örnek talep üzerine doğar — önceden kaydedilmez</text></svg>
</div>

<sub>▲ AgentId = tip + anahtar · düzenlemek için: [`f_identity.excalidraw`](diagrams/wiki/f_identity.excalidraw) → excalidraw.com'a sürükle</sub>


`AgentId(type, key)` — **iki parçalı**. Ve en az konuşulan, en çok işe yarayan
mekanizma şu:

> **Topic kaynağı, ajan anahtarına dönüşüyor.**
> `TopicId("tur", "oturum-42")`'ye yayın yapmak `AgentId("session", "oturum-42")`
> ajanını **yaratıyor** — oturum başına izole örnek, elle sözlük tutmadan.

Bu projede gateway oturumları tam olarak böyle çalışıyor. Ölçek gerektiğinde
ilk bakılacak yer burası.

---

<a id="s5"></a>
## 5 · İki iletişim biçimi

<div align="center">
<svg viewBox="0 0 600 154" width="600" height="154"><text x="20" y="18" font-size="8.6" fill="#1971c2" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">send_message — DOĞRUDAN</text><path d="M21,31 L115,31 L115,69 L21,69 Z" fill="#e7f5ff" stroke="none"/><path d="M20.6,30.6 Q68.0,29.1 117.1,29.0 M114.9,29.1 Q116.4,50.0 116.7,68.8 M115.1,69.3 Q68.0,69.9 19.8,71.1 M18.4,69.3 Q20.2,50.0 18.9,31.2" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M21.0,31.0 Q68.0,31.1 116.8,31.4 M115.2,31.1 Q115.5,50.0 116.0,70.8 M115.8,69.6 Q68.0,69.8 19.8,69.4 M21.0,71.0 Q19.4,50.0 21.6,30.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="68.0" y="53.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">gönderen</text><path d="M179,31 L273,31 L273,69 L179,69 Z" fill="#e7f5ff" stroke="none"/><path d="M178.8,28.8 Q226.0,30.2 274.6,29.8 M273.0,30.1 Q274.5,50.0 273.2,69.9 M273.7,68.8 Q226.0,69.6 178.0,69.2 M177.0,69.6 Q177.6,50.0 176.6,30.5" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M177.6,30.7 Q226.0,31.6 275.4,28.5 M275.0,31.1 Q274.6,50.0 274.4,68.9 M275.2,71.2 Q226.0,70.3 179.0,69.0 M177.6,70.7 Q178.6,50.0 176.8,31.5" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="226.0" y="53.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">alıcı</text><path d="M118.7,49.1 Q147.0,50.3 176.4,50.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M117.5,50.7 Q147.0,50.3 175.5,49.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M176.0,50.0 L169.3,54.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M176.0,50.0 L169.2,45.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="147.0" y="44.0" font-size="7.4" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" text-anchor="middle">1 → 1</text><text x="20" y="88" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">· cevabı DÖNDÜRÜR</text><text x="20" y="102" font-size="7.6" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">· hata ÇAĞIRANA fırlar</text><text x="20" y="116" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">· tek alıcı, bilinen adres</text><text x="330" y="18" font-size="8.6" fill="#2f9e44" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">publish_message — YAYIN</text><path d="M331,31 L425,31 L425,69 L331,69 Z" fill="#ebfbee" stroke="none"/><path d="M330.5,30.2 Q378.0,28.4 426.1,28.5 M425.5,28.7 Q426.1,50.0 424.4,71.6 M425.4,69.3 Q378.0,71.4 330.3,68.6 M329.6,71.1 Q330.0,50.0 328.4,29.5" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M331.5,30.1 Q378.0,30.0 425.9,30.5 M424.5,30.0 Q426.3,50.0 427.3,70.8 M426.6,71.1 Q378.0,70.9 329.2,70.3 M329.2,69.3 Q329.8,50.0 329.1,29.5" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="378.0" y="53.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">yayıncı</text><path d="M489,13 L579,13 L579,41 L489,41 Z" fill="#ebfbee" stroke="none"/><path d="M487.7,10.7 Q534.0,10.3 580.8,12.3 M580.9,11.0 Q580.6,27.0 579.9,42.2 M579.3,41.8 Q534.0,42.6 486.8,43.2 M489.0,41.1 Q487.5,27.0 487.4,11.6" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M486.8,11.6 Q534.0,10.3 579.8,13.2 M578.5,12.1 Q579.4,27.0 578.8,43.0 M581.3,42.1 Q534.0,42.6 486.8,40.4 M489.2,40.9 Q487.7,27.0 487.9,11.6" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="534.0" y="30.2" font-size="7.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">abone 1</text><path d="M489,49 L579,49 L579,77 L489,77 Z" fill="#ebfbee" stroke="none"/><path d="M489.0,47.1 Q534.0,47.4 580.7,48.5 M580.1,49.3 Q579.5,63.0 579.8,78.2 M581.1,79.3 Q534.0,77.6 486.8,76.8 M489.0,78.1 Q488.0,63.0 487.4,49.5" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M486.7,46.6 Q534.0,47.3 578.5,48.7 M580.7,49.2 Q580.5,63.0 581.6,79.0 M580.4,79.1 Q534.0,78.3 488.7,77.3 M486.8,77.4 Q488.3,63.0 489.6,47.9" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="534.0" y="66.2" font-size="7.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">abone 2</text><path d="M428.9,43.8 Q456.8,35.3 484.4,26.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M427.5,45.4 Q456.8,35.3 485.2,29.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M486.0,28.0 L481.6,33.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M486.0,28.0 L479.6,26.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M427.2,55.9 Q457.0,58.0 487.3,60.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M427.6,55.9 Q457.1,57.1 485.0,60.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M486.0,60.0 L478.8,64.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M486.0,60.0 L479.6,54.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="330" y="88" font-size="7.6" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">· None DÖNER — cevap yok</text><text x="330" y="102" font-size="7.6" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">· hata yalnız LOGLANIR</text><text x="330" y="116" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">· 0 abone de geçerli bir sonuç</text><text x="20" y="142" font-size="8" fill="#8a5208" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Bu asimetri ölçüldü: yayında düşen bir handler sessizce düşer.</text></svg>
</div>

<sub>▲ Doğrudan ve yayın — asimetri hatada · düzenlemek için: [`f_send_vs_publish.excalidraw`](diagrams/wiki/f_send_vs_publish.excalidraw) → excalidraw.com'a sürükle</sub>


| | `send_message` | `publish_message` |
|---|---|---|
| Alıcı | tek `AgentId` | topic'e abone olan herkes |
| Dönüş | **var** | **yok** |
| Handler çökerse | çağırana **fırlatır** | **loglanır, fırlatmaz** |

Son satır bir tasarım kararı: sonucu bekleyeceksen doğrudan, olay duyuracaksan
yayın.

### Ve buradan doğan en pahalı arıza

<div align="center">
<svg viewBox="0 0 600 172" width="600" height="172"><path d="M21,57 L123,57 L123,99 L21,99 Z" fill="#ffffff" stroke="none"/><path d="M18.4,56.8 Q72.0,57.7 122.9,57.6 M125.2,56.6 Q124.9,78.0 125.1,101.6 M123.5,100.7 Q72.0,100.9 19.3,99.2 M21.1,101.2 Q19.5,78.0 21.0,55.1" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><path d="M20.0,55.1 Q72.0,57.5 123.9,55.7 M124.2,55.6 Q124.7,78.0 124.2,101.4 M123.7,101.2 Q72.0,99.5 21.6,100.0 M18.5,98.8 Q20.8,78.0 19.0,57.4" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><text x="72.0" y="81.2" font-size="8.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">koordinatör</text><path d="M197,9 L311,9 L311,43 L197,43 Z" fill="#e7f5ff" stroke="none"/><path d="M196.7,7.3 Q254.0,8.7 312.0,9.2 M310.8,6.6 Q312.6,26.0 311.9,45.1 M311.7,44.0 Q254.0,43.8 195.0,45.4 M195.2,43.4 Q195.8,26.0 197.0,7.4" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M195.5,7.8 Q254.0,9.5 310.6,9.3 M311.9,6.8 Q311.6,26.0 312.0,43.6 M313.4,44.8 Q254.0,42.8 195.3,42.5 M197.4,42.9 Q196.6,26.0 196.0,7.9" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="254.0" y="25.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">arXiv</text><text x="254.0" y="36.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">RoutedAgent</text><path d="M197,59 L311,59 L311,93 L197,93 Z" fill="#e7f5ff" stroke="none"/><path d="M194.5,56.6 Q254.0,59.7 313.1,58.2 M312.9,58.3 Q312.1,76.0 312.6,93.2 M310.4,93.5 Q254.0,95.5 196.5,93.6 M196.4,94.7 Q195.8,76.0 196.3,57.4" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M197.0,59.3 Q254.0,59.2 313.3,56.4 M311.5,57.6 Q311.8,76.0 312.1,92.9 M313.6,93.5 Q254.0,94.9 197.3,94.6 M195.4,94.3 Q195.7,76.0 196.6,57.5" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="254.0" y="75.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">HN</text><text x="254.0" y="86.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">RoutedAgent</text><path d="M197,109 L311,109 L311,143 L197,143 Z" fill="#e7f5ff" stroke="none"/><path d="M195.0,109.2 Q254.0,106.4 312.7,106.7 M312.2,107.5 Q311.5,126.0 313.3,142.8 M311.5,144.8 Q254.0,145.4 196.2,143.9 M195.7,144.9 Q196.6,126.0 194.4,108.9" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M196.5,109.1 Q254.0,106.8 312.5,108.3 M312.8,107.6 Q311.9,126.0 311.1,144.2 M310.8,144.1 Q254.0,143.9 195.9,142.9 M196.9,143.5 Q196.3,126.0 197.2,109.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="254.0" y="125.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">GitHub</text><text x="254.0" y="136.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">RoutedAgent</text><path d="M125.7,77.3 Q160.4,52.5 192.8,25.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M124.6,79.3 Q161.3,53.7 194.6,26.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M194.0,26.0 L191.0,33.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M194.0,26.0 L186.1,26.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M126.7,78.1 Q160.0,78.0 195.2,75.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M127.2,76.6 Q160.0,75.9 193.7,75.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M194.0,76.0 L188.4,80.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M194.0,76.0 L187.8,72.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M125.9,77.9 Q160.0,101.9 194.3,125.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M127.4,79.5 Q160.2,101.7 193.2,127.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M194.0,126.0 L185.8,125.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M194.0,126.0 L191.9,118.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M393,57 L519,57 L519,99 L393,99 Z" fill="#ebfbee" stroke="none"/><path d="M393.6,56.4 Q456.0,56.6 519.9,54.5 M519.9,56.9 Q520.1,78.0 519.6,101.0 M520.4,100.8 Q456.0,100.3 390.9,101.5 M391.9,101.5 Q391.9,78.0 392.2,56.1" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M391.1,56.8 Q456.0,55.0 520.0,56.5 M520.6,55.4 Q519.7,78.0 521.2,99.8 M521.2,101.3 Q456.0,101.2 390.8,101.2 M391.3,100.2 Q392.3,78.0 393.4,54.9" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="456.0" y="77.2" font-size="8.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">ClosureAgent</text><text x="456.0" y="88.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">toplayıcı</text><path d="M313.2,24.9 Q351.6,52.6 391.4,79.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M314.9,25.7 Q351.6,52.6 389.8,78.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M390.0,78.0 L382.9,78.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M390.0,78.0 L387.2,71.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M315.0,77.2 Q352.0,76.6 390.9,77.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M315.5,75.6 Q352.0,76.6 390.5,76.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M390.0,78.0 L383.2,81.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M390.0,78.0 L384.1,73.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M312.6,126.6 Q352.4,102.6 390.6,77.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M313.2,126.0 Q351.4,101.0 389.0,77.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M390.0,78.0 L387.4,85.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M390.0,78.0 L382.0,77.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="196" y="162" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">hepsi AYNI topic'e abone — tek publish üçünü birden uyandırır</text><text x="392" y="116" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">sayaç 3'e ulaşınca</text><text x="392" y="130" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">kuyruğa yazar</text><text x="20" y="116" font-size="7.6" fill="#2f9e44" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">1 publish</text><text x="20" y="130" font-size="7.6" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">0 dönüş değeri</text></svg>
</div>

<sub>▲ Fan-out / fan-in — ve sessiz kayıp · düzenlemek için: [`f_fanout.excalidraw`](diagrams/wiki/f_fanout.excalidraw) → excalidraw.com'a sürükle</sub>


Çöken bir handler `_process_publish` içindeki `gather`'ı erken döndürüyor,
`stop_when_idle()` bariyeri erken açılıyor, ve **tamamlanmış kardeş sonuçlar
sessizce kayboluyor.**

Aynı arıza enjeksiyonuyla ölçüldü **[ölçüldü]**:

| Motor | Temiz | Sarmalayıcı arkasında | Ham hata |
|---|---:|---:|---:|
| GraphFlow | 3 | 2 | **0–1**, süre sınırı dolar |
| core pub/sub + `ClosureAgent` kuyruğu | 3 | 2 | **2**, ~3 ms |

> Resmî desenler bu konuda **birbiriyle çelişiyor**: *Concurrent Agents* kuyrukla
> topluyor, *Mixture of Agents* `asyncio.gather(...)` ile — sessiz kaybın kaynağı
> olan yapı.

---

<a id="s6"></a>
## 6 · Tool döngüsü

<div align="center">
<svg viewBox="0 0 600 152" width="600" height="152"><path d="M17,53 L119,53 L119,95 L17,95 Z" fill="#ffffff" stroke="none"/><path d="M16.6,52.4 Q68.0,50.8 119.9,51.1 M121.0,52.0 Q119.5,74.0 120.0,95.2 M119.6,96.3 Q68.0,97.8 14.6,96.9 M15.1,94.5 Q15.5,74.0 17.6,52.8" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><path d="M16.4,50.5 Q68.0,53.4 119.5,52.0 M121.4,51.6 Q120.7,74.0 118.9,97.0 M121.4,95.9 Q68.0,97.4 16.2,95.5 M15.0,94.5 Q16.0,74.0 17.2,51.2" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><text x="68.0" y="77.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">AssistantAgent</text><path d="M177,53 L279,53 L279,95 L177,95 Z" fill="#e7f5ff" stroke="none"/><path d="M175.1,53.5 Q228.0,53.0 281.3,52.8 M280.3,52.7 Q279.5,74.0 278.8,95.9 M279.0,97.4 Q228.0,94.8 177.2,95.7 M175.8,95.6 Q175.8,74.0 177.6,51.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M174.7,52.7 Q228.0,50.3 281.6,52.0 M280.2,52.8 Q280.0,74.0 279.1,96.3 M280.4,95.0 Q228.0,95.8 174.9,95.5 M175.5,96.3 Q176.8,74.0 175.4,50.7" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="228.0" y="73.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">model</text><text x="228.0" y="84.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">create_stream</text><path d="M337,53 L439,53 L439,95 L337,95 Z" fill="#fff4e6" stroke="none"/><path d="M337.0,52.9 Q388.0,50.4 439.6,52.6 M441.3,53.5 Q439.7,74.0 439.7,97.6 M440.3,97.2 Q388.0,94.3 336.0,95.6 M337.2,94.9 Q335.6,74.0 334.6,53.0" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><path d="M335.9,53.1 Q388.0,53.0 438.6,51.2 M439.7,50.9 Q440.4,74.0 438.6,95.5 M441.5,96.8 Q388.0,94.6 335.9,95.2 M336.1,96.7 Q336.8,74.0 334.5,51.1" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><text x="388.0" y="73.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">workbench</text><text x="388.0" y="84.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">call_tool</text><path d="M487,53 L581,53 L581,95 L487,95 Z" fill="#ebfbee" stroke="none"/><path d="M486.9,53.2 Q534.0,53.2 582.3,51.8 M583.1,51.4 Q581.9,74.0 582.1,95.3 M580.4,96.6 Q534.0,95.0 487.5,94.5 M486.5,97.1 Q486.2,74.0 486.3,51.2" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M487.5,53.3 Q534.0,51.7 580.8,50.7 M580.5,51.8 Q582.7,74.0 580.9,97.2 M580.8,96.1 Q534.0,95.4 485.4,96.0 M484.9,95.0 Q485.5,74.0 485.6,52.7" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="534.0" y="77.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">sonuç</text><path d="M120.5,74.7 Q148.0,74.1 175.2,74.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M120.4,73.9 Q148.0,74.7 172.6,74.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M174.0,74.0 L167.3,78.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M174.0,74.0 L167.4,70.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M280.5,73.6 Q308.0,73.5 333.4,75.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M282.0,74.2 Q308.0,73.5 335.4,72.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M334.0,74.0 L327.9,78.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M334.0,74.0 L328.1,70.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="308.0" y="68.0" font-size="7.4" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" text-anchor="middle">tool isteği</text><path d="M442.6,74.1 Q463.0,73.6 484.2,72.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M443.5,73.5 Q463.0,74.6 483.4,75.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M484.0,74.0 L477.2,77.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M484.0,74.0 L477.2,70.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M535.8,52.4 Q518.5,33.5 498.5,16.5 M500.1,15.6 Q400.0,12.3 299.7,7.1 M299.8,7.2 Q210.2,17.2 120.7,23.0 M119.9,21.2 Q93.7,35.4 67.6,50.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M537.5,50.7 Q518.3,33.7 499.9,16.0 M501.2,16.9 Q400.0,11.0 300.4,8.8 M301.1,8.6 Q209.9,14.2 120.6,21.2 M119.6,22.4 Q93.4,34.8 69.0,51.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M68.0,50.0 L72.2,42.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M68.0,50.0 L75.2,50.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="280" y="6" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">döngü — max_tool_iterations</text><text x="16" y="122" font-size="8.4" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">VARSAYILAN 1: model tool sonucunu GÖRMEDEN cevap verir</text><text x="16" y="138" font-size="7.6" fill="#2f9e44" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">ölçüldü — bizde 6'ya çekildi</text></svg>
</div>

<sub>▲ Model ister · kapı · çalışır · sonucu görür · döngü · düzenlemek için: [`f_tool_loop.excalidraw`](diagrams/wiki/f_tool_loop.excalidraw) → excalidraw.com'a sürükle</sub>


### Tarif = arayüz

Model fonksiyonu görmüyor; **adını, tarifini ve parametre şemasını** görüyor.
`description` prompt'a giren tek metin, ve modelin o tool'a *ne zaman*
uzanacağına karar verdiği şey o.

### Varsayılan tavan — altı çerçeve, altı cevap

Hepsi kurulu paketten okundu **[ölçüldü]**:

| Çerçeve | Alan | Varsayılan |
|---|---|---:|
| **AutoGen** | `max_tool_iterations` | **1** |
| OpenAI Agents SDK | `Runner.run(max_turns=)` | 10 |
| CrewAI | `Agent.max_iter` | 25 |
| **MAF** | `DEFAULT_MAX_ITERATIONS` | **40** |
| LangGraph | `recursion_limit` | 10007 |
| Google ADK | `LoopAgent.max_iterations` | sınırsız |

AutoGen'de **1**: ajan tool'u çağırır, sonucu görür, **durur** — ve hata vermez.
Microsoft bunu göç kılavuzunda kendisi yazıyor **[kaynak]**.

---

<a id="s7"></a>
## 7 · Beş takım

<div align="center">
<svg viewBox="0 0 600 128" width="600" height="128"><path d="M13,27 L115,27 L115,77 L13,77 Z" fill="#f8f0fc" stroke="none"/><path d="M10.8,24.5 Q64.0,27.6 117.6,25.0 M116.5,25.5 Q116.8,52.0 117.2,77.1 M115.4,78.3 Q64.0,76.3 13.4,78.6 M12.7,76.6 Q12.9,52.0 13.2,26.3" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M11.0,27.1 Q64.0,25.3 116.3,27.4 M117.5,26.2 Q115.4,52.0 116.6,78.4 M117.5,77.4 Q64.0,78.9 12.9,79.5 M10.9,79.0 Q12.3,52.0 12.5,25.0" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="64.0" y="51.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">RoundRobin</text><text x="64.0" y="62.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">sırayla</text><path d="M131,27 L233,27 L233,77 L131,77 Z" fill="#f8f0fc" stroke="none"/><path d="M130.1,24.5 Q182.0,24.4 235.4,26.0 M233.7,26.5 Q234.0,52.0 235.3,77.4 M234.3,77.4 Q182.0,78.5 130.3,78.1 M130.6,78.1 Q129.9,52.0 128.7,27.6" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M129.8,25.7 Q182.0,25.5 235.4,27.4 M233.7,24.7 Q234.0,52.0 235.3,76.9 M233.5,77.5 Q182.0,77.8 129.5,77.0 M130.1,78.9 Q129.0,52.0 130.5,25.0" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="182.0" y="51.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Selector</text><text x="182.0" y="62.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">model seçer</text><path d="M249,27 L351,27 L351,77 L249,77 Z" fill="#f8f0fc" stroke="none"/><path d="M246.6,25.7 Q300.0,26.7 353.0,27.3 M353.1,26.0 Q352.1,52.0 352.1,78.3 M352.8,77.9 Q300.0,76.2 248.7,78.5 M248.9,79.0 Q247.7,52.0 248.3,27.4" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M248.0,26.4 Q300.0,24.3 351.5,24.9 M351.8,25.3 Q351.6,52.0 350.5,79.5 M350.8,77.1 Q300.0,78.5 247.2,79.2 M247.3,76.7 Q247.3,52.0 247.9,25.1" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="300.0" y="51.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Swarm</text><text x="300.0" y="62.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">handoff</text><path d="M367,27 L469,27 L469,77 L367,77 Z" fill="#f8f0fc" stroke="none"/><path d="M366.8,27.4 Q418.0,25.4 471.1,24.5 M471.1,24.5 Q470.3,52.0 469.1,78.0 M470.8,78.7 Q418.0,78.4 365.9,77.2 M365.3,76.6 Q365.4,52.0 366.5,24.9" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M366.5,24.6 Q418.0,27.4 470.0,26.3 M470.0,24.5 Q471.0,52.0 470.9,76.4 M471.6,79.1 Q418.0,77.5 365.1,78.1 M365.5,79.2 Q365.3,52.0 367.0,24.4" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="418.0" y="51.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">MagenticOne</text><text x="418.0" y="62.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">planlayıcı</text><path d="M485,27 L587,27 L587,77 L485,77 Z" fill="#f8f0fc" stroke="none"/><path d="M484.4,25.0 Q536.0,25.6 589.0,27.3 M587.6,27.1 Q587.9,52.0 588.0,77.0 M586.9,79.1 Q536.0,78.8 484.0,78.6 M485.5,77.0 Q484.7,52.0 483.7,27.2" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M483.6,26.5 Q536.0,26.9 587.0,24.9 M587.6,25.2 Q588.9,52.0 588.1,79.5 M588.2,79.2 Q536.0,79.2 484.3,78.1 M482.5,79.4 Q484.6,52.0 484.6,27.0" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="536.0" y="51.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">GraphFlow</text><text x="536.0" y="62.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">DAG</text><text x="12" y="100" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Beşi de aynı arayüz: run() / run_stream() → TaskResult</text><text x="12" y="116" font-size="7.6" fill="#2f9e44" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Taramamız GraphFlow kullanıyor — eşzamanlı dal + join(all)</text></svg>
</div>

<sub>▲ Değişen tek şey: sırayı kim belirliyor · düzenlemek için: [`f_teams.excalidraw`](diagrams/wiki/f_teams.excalidraw) → excalidraw.com'a sürükle</sub>


Aynı görev, aynı ajanlar **[ölçüldü]**:

| Desen | Sırayı kim belirliyor | Mesaj | LLM | Tool | Token |
|---|---|---:|---:|---:|---:|
| **SelectorGroupChat** | model her turda seçiyor | 8 | 5 | 2 | **204** |
| GraphFlow | önceden çizilmiş DAG | 11 | 7 | 3 | 270 |
| RoundRobinGroupChat | sırayla | 9 | 6 | 2 | 274 |
| **Swarm** | ajan devrediyor | 14 | 7 | 4 | **334** |

**%63,7 fark.** Ödenen zekâ değil **yönlendirme özerkliği**.

### GraphFlow — boruyu çizmek

<div align="center">
<svg viewBox="0 0 600 182" width="600" height="182"><path d="M17,61 L107,61 L107,99 L17,99 Z" fill="#ffffff" stroke="none"/><path d="M17.4,61.3 Q62.0,61.5 106.7,60.3 M108.1,58.8 Q108.1,80.0 107.0,99.8 M107.9,98.5 Q62.0,101.0 14.7,100.7 M16.0,100.7 Q15.9,80.0 15.5,58.6" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><path d="M16.3,60.5 Q62.0,61.0 108.4,61.5 M108.8,59.6 Q108.2,80.0 108.2,100.5 M106.7,99.9 Q62.0,100.7 16.7,100.3 M16.5,99.0 Q15.9,80.0 15.0,59.4" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><text x="62.0" y="83.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">giriş</text><path d="M169,11 L271,11 L271,45 L169,45 Z" fill="#e7f5ff" stroke="none"/><path d="M167.0,8.8 Q220.0,11.1 270.7,8.5 M272.2,11.0 Q272.3,28.0 271.5,45.6 M271.4,46.9 Q220.0,46.8 168.2,45.2 M168.9,46.7 Q167.4,28.0 169.1,9.2" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M169.3,10.3 Q220.0,10.2 273.4,8.9 M270.6,9.7 Q271.3,28.0 270.5,47.0 M273.5,47.0 Q220.0,45.3 169.4,46.1 M168.8,45.5 Q168.7,28.0 169.5,9.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="220.0" y="31.2" font-size="7.8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">analist A</text><path d="M169,63 L271,63 L271,97 L169,97 Z" fill="#e7f5ff" stroke="none"/><path d="M166.6,61.9 Q220.0,60.9 271.0,63.4 M271.1,60.7 Q271.4,80.0 272.1,96.9 M273.0,97.2 Q220.0,97.8 167.7,99.3 M167.5,99.3 Q167.9,80.0 169.0,63.3" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M167.7,62.6 Q220.0,62.7 273.1,62.4 M271.3,61.7 Q272.4,80.0 272.9,98.2 M273.0,96.7 Q220.0,97.8 168.4,97.5 M169.1,97.2 Q168.1,80.0 169.4,60.7" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="220.0" y="83.2" font-size="7.8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">analist B</text><path d="M169,115 L271,115 L271,149 L169,149 Z" fill="#e7f5ff" stroke="none"/><path d="M168.1,113.2 Q220.0,114.2 272.2,113.9 M272.7,112.4 Q272.1,132.0 271.6,150.9 M273.2,149.1 Q220.0,151.3 168.6,149.0 M167.8,148.7 Q167.9,132.0 168.6,113.9" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M168.1,114.2 Q220.0,114.7 272.1,113.4 M270.7,115.2 Q271.4,132.0 273.2,150.0 M272.6,150.3 Q220.0,149.0 167.3,149.4 M167.4,151.5 Q167.8,132.0 167.5,113.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="220.0" y="135.2" font-size="7.8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">analist C</text><path d="M111.0,79.9 Q137.8,53.8 166.2,27.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M108.7,79.1 Q137.4,53.4 165.2,29.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M166.0,28.0 L164.3,34.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M166.0,28.0 L158.5,28.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M109.8,81.1 Q138.0,80.2 165.2,81.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M111.5,80.4 Q138.0,79.3 164.5,81.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M166.0,80.0 L159.5,84.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M166.0,80.0 L159.3,76.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M110.8,79.7 Q139.0,104.9 166.6,131.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M111.3,80.5 Q139.1,104.8 164.9,133.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M166.0,132.0 L159.2,130.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M166.0,132.0 L163.6,124.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M337,59 L439,59 L439,101 L337,101 Z" fill="#fff4e6" stroke="none"/><path d="M336.7,57.9 Q388.0,56.7 440.7,58.2 M441.0,58.3 Q439.5,80.0 440.3,101.4 M438.7,103.1 Q388.0,103.3 335.1,101.8 M336.4,102.1 Q335.7,80.0 337.1,57.8" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><path d="M336.5,58.8 Q388.0,57.5 441.4,57.4 M441.5,59.5 Q439.3,80.0 438.8,101.6 M441.1,100.5 Q388.0,101.0 335.7,102.7 M336.1,101.1 Q336.5,80.0 335.7,56.6" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><text x="388.0" y="79.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">join(all)</text><text x="388.0" y="90.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">hepsini bekler</text><path d="M273.8,29.0 Q304.5,53.5 334.3,79.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M273.4,28.5 Q304.6,53.3 333.8,79.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M334.0,80.0 L327.0,79.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M334.0,80.0 L332.7,72.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M273.8,80.3 Q304.0,80.4 335.3,81.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M273.8,78.6 Q304.0,80.5 335.1,79.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M334.0,80.0 L327.6,83.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M334.0,80.0 L327.1,75.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M274.8,132.7 Q305.2,107.4 334.5,79.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M274.3,130.6 Q303.2,105.0 334.3,80.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M334.0,80.0 L332.1,86.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M334.0,80.0 L326.0,80.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M491,59 L585,59 L585,101 L491,101 Z" fill="#ebfbee" stroke="none"/><path d="M490.5,57.4 Q538.0,56.7 586.4,58.8 M586.1,57.0 Q585.8,80.0 585.5,103.5 M585.3,102.0 Q538.0,102.8 490.1,102.9 M491.5,100.5 Q489.6,80.0 490.9,58.2" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M488.9,58.6 Q538.0,59.4 587.4,58.5 M587.3,59.5 Q586.7,80.0 584.8,103.0 M584.7,102.9 Q538.0,103.5 490.7,103.5 M490.0,101.3 Q490.2,80.0 490.2,59.6" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="538.0" y="83.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">sayım</text><path d="M443.2,79.0 Q465.0,80.3 488.8,80.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M443.0,79.0 Q465.0,79.0 487.1,78.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M488.0,80.0 L482.0,83.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M488.0,80.0 L482.4,75.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="168" y="170" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">DiGraphBuilder ile kurulur · dallar EŞZAMANLI koşar</text><text x="336" y="118" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">join politikası: all | any</text></svg>
</div>

<sub>▲ DiGraphBuilder ile akış · düzenlemek için: [`f_graphflow.excalidraw`](diagrams/wiki/f_graphflow.excalidraw) → excalidraw.com'a sürükle</sub>


Kenarlar **veri taşımıyor**, yalnız sırayı belirliyor. Join'de
`activation_condition="all"` demezsen ilk gelen dal akışı ilerletiyor.

---

<a id="s8"></a>
## 8 · Durmayı öğretmek

<div align="center">
<svg viewBox="0 0 600 186" width="600" height="186"><path d="M17,23 L273,23 L273,117 L17,117 Z" fill="#ebfbee" stroke="none"/><path d="M16.0,22.0 Q145.0,20.7 275.5,23.0 M274.6,20.9 Q273.0,70.0 275.3,117.8 M272.7,117.4 Q145.0,117.7 17.1,116.8 M15.4,117.5 Q14.5,70.0 16.0,22.0" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M14.4,21.5 Q145.0,20.9 273.4,22.1 M274.7,21.8 Q273.8,70.0 273.6,118.5 M274.5,116.5 Q145.0,116.5 15.4,119.5 M15.4,117.1 Q17.7,70.0 14.4,20.8" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="145.0" y="73.2" font-size="9.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle"></text><text x="145" y="42" font-size="9" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">SERT TAVAN</text><text x="30" y="62" font-size="7.4" fill="#454c53" font-family="DejaVu Sans Mono, monospace">MaxMessageTermination(20)</text><text x="30" y="79" font-size="7.4" fill="#454c53" font-family="DejaVu Sans Mono, monospace">TokenUsageTermination(50_000)</text><text x="30" y="96" font-size="7.4" fill="#454c53" font-family="DejaVu Sans Mono, monospace">TimeoutTermination(300)</text><text x="30" y="112" font-size="7" fill="#2f9e44" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">modelden bağımsız — her zaman tutar</text><path d="M327,23 L583,23 L583,117 L327,117 Z" fill="#fff5f5" stroke="none"/><path d="M327.5,23.4 Q455.0,20.5 583.5,20.5 M584.4,23.3 Q582.5,70.0 583.4,119.2 M584.5,118.5 Q455.0,116.9 326.6,117.9 M324.7,117.7 Q327.6,70.0 325.6,21.9" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><path d="M324.6,21.7 Q455.0,23.2 583.8,21.6 M584.6,22.4 Q583.6,70.0 584.0,117.6 M585.0,118.5 Q455.0,116.2 324.7,117.6 M325.0,117.8 Q326.7,70.0 325.8,20.8" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><text x="455.0" y="73.2" font-size="9.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle"></text><text x="455" y="42" font-size="9" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">ANLAMSAL KOŞUL</text><text x="340" y="62" font-size="7.4" fill="#454c53" font-family="DejaVu Sans Mono, monospace">TextMentionTermination("BİTTİ")</text><text x="340" y="79" font-size="7.4" fill="#454c53" font-family="DejaVu Sans Mono, monospace">HandoffTermination(...)</text><text x="340" y="96" font-size="7.4" fill="#454c53" font-family="DejaVu Sans Mono, monospace">FunctionCallTermination(...)</text><text x="340" y="112" font-size="7" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">modelin işbirliğine bağlı — yazmazsa hiç tetiklenmez</text><text x="150" y="140" font-size="7.6" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">termination = MaxMessageTermination(20) | TokenUsageTermination(50_000)</text><text x="16" y="160" font-size="7.6" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">| → biri yeterli   ·   &amp; → hepsi gerekli</text><text x="16" y="174" font-size="7.6" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Üretimde en az bir SERT tavan olmadan koşmak, faturayı modelin kararına bırakmaktır.</text></svg>
</div>

<sub>▲ On bir sonlandırma koşulu · düzenlemek için: [`f_termination.excalidraw`](diagrams/wiki/f_termination.excalidraw) → excalidraw.com'a sürükle</sub>


Sonlandırma koşulu olmayan takım **sonsuza kadar** konuşuyor, ve fatura gerçek.
On bir koşul var; en çok kullanılan dördü:

* `MaxMessageTermination` — mesaj sayar
* `TokenUsageTermination` — **token sayar**, faturaya en yakın olan
* `TimeoutTermination` — süre
* `TextMentionTermination` — bir kelime geçince

> Koşullar `&` ve `|` ile birleşiyor. Yalnız mesaj sayan bir koşul, uzun
> cevaplarla dolu bir turu ucuz sanıyor.

---

<a id="s9"></a>
## 9 · Sekiz desen

<div align="center">
<svg viewBox="0 0 600 168" width="600" height="168"><path d="M13,7 L587,7 L587,19 L13,19 Z" fill="#e7f5ff" stroke="none"/><path d="M12.5,6.4 Q300.0,5.9 588.6,7.3 M586.9,5.1 Q587.9,13.0 587.9,19.6 M588.7,19.9 Q300.0,18.3 11.5,19.1 M13.4,18.9 Q12.1,13.0 11.8,5.5" fill="none" stroke="#1971c2" stroke-width="1.1" stroke-linecap="round"/><path d="M11.2,4.6 Q300.0,6.4 588.6,5.4 M589.3,6.3 Q587.9,13.0 589.3,21.0 M587.2,20.5 Q300.0,21.6 11.9,18.4 M12.5,19.8 Q12.2,13.0 12.1,4.9" fill="none" stroke="#1971c2" stroke-width="1.1" stroke-linecap="round"/><text x="22" y="16" font-size="7.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Concurrent Agents</text><text x="180" y="16" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:3236</text><text x="248" y="16" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">tek yayın → çok dal → toplayıcı</text><path d="M13,24 L587,24 L587,36 L13,36 Z" fill="#ebfbee" stroke="none"/><path d="M10.9,23.5 Q300.0,23.2 589.4,22.1 M588.2,24.2 Q588.1,30.0 586.9,36.7 M586.9,35.6 Q300.0,36.6 12.8,37.4 M12.1,35.7 Q12.0,30.0 11.9,23.0" fill="none" stroke="#2f9e44" stroke-width="1.1" stroke-linecap="round"/><path d="M10.9,21.6 Q300.0,23.1 588.1,23.6 M589.4,22.6 Q587.9,30.0 587.7,36.5 M588.5,38.2 Q300.0,37.0 10.7,37.2 M12.8,36.8 Q12.1,30.0 10.6,22.4" fill="none" stroke="#2f9e44" stroke-width="1.1" stroke-linecap="round"/><text x="22" y="33" font-size="7.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Sequential Workflow</text><text x="180" y="33" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:3504</text><text x="248" y="33" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">her ajan bir sonrakine devrediyor</text><path d="M13,41 L587,41 L587,53 L13,53 Z" fill="#f8f0fc" stroke="none"/><path d="M12.9,39.1 Q300.0,40.3 587.4,39.6 M589.5,39.5 Q588.3,47.0 587.8,55.5 M587.3,55.4 Q300.0,53.0 13.0,54.2 M13.1,55.3 Q12.0,47.0 12.5,39.4" fill="none" stroke="#5f3dc4" stroke-width="1.1" stroke-linecap="round"/><path d="M12.5,41.4 Q300.0,41.3 589.0,40.6 M587.8,40.3 Q587.8,47.0 588.8,53.8 M589.1,54.1 Q300.0,55.3 11.9,54.1 M13.1,53.3 Q11.9,47.0 11.2,40.4" fill="none" stroke="#5f3dc4" stroke-width="1.1" stroke-linecap="round"/><text x="22" y="50" font-size="7.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Group Chat</text><text x="180" y="50" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:3772</text><text x="248" y="50" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">bir yönetici konuşma sırasını dağıtıyor</text><path d="M13,58 L587,58 L587,70 L13,70 Z" fill="#fff4e6" stroke="none"/><path d="M11.4,55.7 Q300.0,57.0 587.8,56.7 M588.5,56.3 Q588.0,64.0 586.6,69.6 M589.4,71.8 Q300.0,70.8 13.4,69.9 M12.7,70.7 Q11.8,64.0 11.7,58.1" fill="none" stroke="#e8590c" stroke-width="1.1" stroke-linecap="round"/><path d="M13.4,56.0 Q300.0,55.5 588.3,57.5 M588.7,56.4 Q588.0,64.0 587.1,71.5 M586.6,69.7 Q300.0,69.4 13.6,69.6 M12.0,70.4 Q11.8,64.0 12.7,56.9" fill="none" stroke="#e8590c" stroke-width="1.1" stroke-linecap="round"/><text x="22" y="67" font-size="7.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Handoffs</text><text x="180" y="67" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:4349</text><text x="248" y="67" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">ajan işi kendisi devrediyor</text><path d="M13,75 L587,75 L587,87 L13,87 Z" fill="#e7f5ff" stroke="none"/><path d="M10.8,73.1 Q300.0,75.1 587.0,75.0 M587.6,73.1 Q587.7,81.0 586.6,87.0 M589.4,89.4 Q300.0,86.3 11.0,86.6 M12.7,89.2 Q12.3,81.0 13.3,75.2" fill="none" stroke="#1971c2" stroke-width="1.1" stroke-linecap="round"/><path d="M13.2,74.2 Q300.0,72.2 589.5,73.0 M586.8,74.8 Q588.1,81.0 588.3,89.2 M588.9,87.6 Q300.0,87.8 10.6,86.6 M10.5,88.7 Q11.8,81.0 12.7,75.6" fill="none" stroke="#1971c2" stroke-width="1.1" stroke-linecap="round"/><text x="22" y="84" font-size="7.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Mixture of Agents</text><text x="180" y="84" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:4989</text><text x="248" y="84" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">aynı soru, farklı uzmanlar, birleştirici</text><path d="M13,92 L587,92 L587,104 L13,104 Z" fill="#f8f0fc" stroke="none"/><path d="M11.0,90.0 Q300.0,92.0 587.6,90.5 M588.9,91.7 Q588.2,98.0 587.7,105.2 M588.2,105.3 Q300.0,104.6 12.4,106.6 M10.7,103.8 Q11.9,98.0 12.3,91.0" fill="none" stroke="#5f3dc4" stroke-width="1.1" stroke-linecap="round"/><path d="M12.1,92.3 Q300.0,92.5 589.1,90.5 M588.5,90.7 Q588.1,98.0 586.9,103.5 M589.5,105.7 Q300.0,103.8 10.8,103.4 M11.7,106.4 Q12.1,98.0 11.9,91.6" fill="none" stroke="#5f3dc4" stroke-width="1.1" stroke-linecap="round"/><text x="22" y="101" font-size="7.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Multi-Agent Debate</text><text x="180" y="101" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:5358</text><text x="248" y="101" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">birden çok tur karşılıklı eleştiri</text><path d="M13,109 L587,109 L587,121 L13,121 Z" fill="#ebfbee" stroke="none"/><path d="M10.5,108.8 Q300.0,106.3 588.4,109.2 M588.2,107.9 Q588.1,115.0 588.9,122.3 M586.9,122.5 Q300.0,120.8 11.6,122.9 M10.9,120.4 Q11.9,115.0 12.0,106.8" fill="none" stroke="#2f9e44" stroke-width="1.1" stroke-linecap="round"/><path d="M13.2,107.4 Q300.0,108.9 588.8,108.6 M586.7,109.5 Q588.0,115.0 587.4,120.6 M589.5,123.0 Q300.0,121.0 12.5,121.5 M11.9,122.7 Q12.3,115.0 12.9,107.7" fill="none" stroke="#2f9e44" stroke-width="1.1" stroke-linecap="round"/><text x="22" y="118" font-size="7.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Reflection</text><text x="180" y="118" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:5822</text><text x="248" y="118" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">üretici + eleştirmen, kalite döngüsü</text><path d="M13,126 L587,126 L587,138 L13,138 Z" fill="#fff5f5" stroke="none"/><path d="M11.7,125.6 Q300.0,126.8 587.1,124.2 M586.9,125.6 Q588.0,132.0 587.0,138.4 M589.2,140.4 Q300.0,137.6 12.9,138.4 M12.9,139.2 Q11.9,132.0 13.4,126.4" fill="none" stroke="#c92a2a" stroke-width="1.1" stroke-linecap="round"/><path d="M10.9,126.2 Q300.0,124.8 589.0,125.1 M588.5,125.0 Q587.9,132.0 587.3,138.7 M589.2,140.3 Q300.0,137.6 13.4,139.0 M13.2,137.4 Q11.9,132.0 13.2,123.5" fill="none" stroke="#c92a2a" stroke-width="1.1" stroke-linecap="round"/><text x="22" y="135" font-size="7.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Code Execution</text><text x="180" y="135" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:6188</text><text x="248" y="135" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">modelin yazdığı kod yürütücüde koşuyor</text><text x="12" y="158" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Bu, core kılavuzunun KENDİ bölümlemesi — bizim tasnifimiz değil. Satır numarası verilmesinin sebebi bu.</text></svg>
</div>

<sub>▲ Resmî sekiz orkestrasyon deseni · düzenlemek için: [`f_patterns.excalidraw`](diagrams/wiki/f_patterns.excalidraw) → excalidraw.com'a sürükle</sub>


Eşzamanlı · Sıralı · Group Chat · Handoff · Mixture of Agents · Münazara ·
Yansıma · Kod yürütme.

Bunlar **kütüphane değil, tarif**. Hiçbiri `import` edilmiyor; kılavuzda kodla
anlatılan yapılar.

---

<a id="s10"></a>
## 10 · Built-in tool'lar — ve neden yok

<div align="center">
<svg viewBox="0 0 600 150" width="600" height="150"><path d="M15,25 L145,25 L145,69 L15,69 Z" fill="#f8f9fa" stroke="none"/><path d="M14.3,23.7 Q80.0,22.5 147.3,24.4 M147.4,23.3 Q146.0,47.0 147.1,69.8 M147.5,71.0 Q80.0,69.4 14.3,71.0 M13.8,71.2 Q13.1,47.0 14.8,23.8" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><path d="M13.5,23.8 Q80.0,23.0 144.8,25.4 M146.1,23.0 Q145.7,47.0 145.4,69.7 M145.5,69.6 Q80.0,71.3 14.8,71.3 M12.8,71.1 Q13.1,47.0 12.4,23.2" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><text x="80.0" y="46.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Python fonksiyonu</text><text x="80.0" y="57.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">tip ipuçları + docstring</text><path d="M148.1,46.1 Q166.0,47.1 184.8,47.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M148.7,46.6 Q166.0,46.1 184.0,47.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M184.0,47.0 L178.0,51.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M184.0,47.0 L177.3,42.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M187,17 L325,17 L325,77 L187,77 Z" fill="#ebfbee" stroke="none"/><path d="M185.0,16.9 Q256.0,14.9 327.2,15.6 M326.9,17.2 Q325.5,47.0 326.9,78.3 M325.4,78.3 Q256.0,76.9 185.3,77.7 M186.0,76.4 Q185.7,47.0 185.8,16.0" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M186.7,14.9 Q256.0,17.5 325.7,16.8 M327.2,14.6 Q327.0,47.0 326.6,78.2 M325.6,77.4 Q256.0,76.4 185.0,77.3 M186.3,77.3 Q185.5,47.0 185.0,17.2" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="256.0" y="46.2" font-size="8.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">FunctionTool</text><text x="256.0" y="57.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">şema TÜRETİLİYOR</text><path d="M329.5,46.2 Q346.0,46.2 364.8,47.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M328.9,45.4 Q346.0,47.8 364.8,48.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M364.0,47.0 L358.2,51.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M364.0,47.0 L357.1,42.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M367,25 L473,25 L473,69 L367,69 Z" fill="#e7f5ff" stroke="none"/><path d="M366.7,22.9 Q420.0,23.1 474.2,24.8 M474.7,25.0 Q473.9,47.0 474.7,68.4 M474.2,71.0 Q420.0,69.4 365.2,68.9 M365.8,68.9 Q365.2,47.0 366.9,24.3" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M365.5,22.7 Q420.0,23.6 472.8,25.2 M474.1,24.2 Q474.6,47.0 474.8,68.9 M472.8,68.8 Q420.0,70.4 365.6,70.1 M365.6,70.2 Q365.7,47.0 364.5,25.1" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="420.0" y="46.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">modele giden</text><text x="420.0" y="57.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">JSON şema</text><path d="M475.7,47.7 Q494.0,46.2 510.4,45.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M474.8,47.3 Q494.0,46.2 512.8,48.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M512.0,47.0 L505.8,51.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M512.0,47.0 L505.7,42.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M515,25 L587,25 L587,69 L515,69 Z" fill="#fff4e6" stroke="none"/><path d="M512.5,25.3 Q551.0,25.4 587.4,23.8 M587.8,25.3 Q587.2,47.0 588.5,68.9 M586.5,70.6 Q551.0,68.6 513.4,71.5 M513.0,69.2 Q513.5,47.0 514.7,23.7" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><path d="M515.3,24.7 Q551.0,24.3 589.0,23.3 M588.0,23.0 Q588.8,47.0 588.8,70.2 M587.1,69.7 Q551.0,69.7 515.1,70.4 M514.2,70.1 Q513.7,47.0 515.1,24.4" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><text x="551.0" y="50.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">çağrı</text><text x="186" y="96" font-size="7.4" fill="#2f9e44" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">docstring = modelin NE ZAMAN çağıracağına karar verdiği metin</text><text x="186" y="110" font-size="7.4" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">tip ipucu yoksa model ne göndereceğini bilemiyor</text><text x="14" y="138" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Tool bir kod parçası; ajan onu modelin ürettiği fonksiyon çağrısına karşılık koşturuyor. Yazdığın açıklama, arayüzün kendisi.</text></svg>
</div>

<sub>▲ Tool: fonksiyon + şema · düzenlemek için: [`f_tools_component.excalidraw`](diagrams/wiki/f_tools_component.excalidraw) → excalidraw.com'a sürükle</sub>


En sık yanılınan yer. **AutoGen hazır tool ile gelmiyor.** `autogen_ext.tools`
altında yedi modül var ve altısı **adaptör** — tool değil **[ölçüldü]**:

| Modül | Ne veriyor | Kurulu mu |
|---|---|---|
| `code_execution` | `PythonCodeExecutionTool` — **tek gerçek tool** | ✔ |
| `mcp` | `StdioMcpToolAdapter` · `SseMcpToolAdapter` · `McpWorkbench` | ✔ |
| `langchain` | `LangChainToolAdapter` — LangChain tool'unu sarmalıyor | ✔ |
| `azure` | Azure AI Search adaptörü | ekstra gerekiyor |
| `graphrag` | GraphRAG adaptörü | ekstra gerekiyor |
| `http` | HTTP çağrısı tool'u | ekstra gerekiyor |
| `semantic_kernel` | SK tool adaptörü | ekstra gerekiyor |

`autogen-ext`'in **38 ayrı ekstrası** var (`docker`, `grpc`, `http-tool`,
`file-surfer`, `jupyter-executor`…). Yetenekler paket içinde değil, **kurulum
seçeneklerinde**.

### Tool'u kim veriyor — üç sistem

| | Hazır tool | Nereden |
|---|---:|---|
| **AutoGen** | **~1** | Kendin yazıyorsun |
| **MAF** | 6 hosted sözleşme | `SupportsCodeInterpreterTool` · `SupportsWebSearchTool` · `SupportsFileSearchTool` · `SupportsImageGenerationTool` · `SupportsShellTool` · `SupportsMCPTool` **[ölçüldü]** |
| **OpenClaw** | **51** (44'ü canlı) | Çekirdekte, 11 grupta |

> **Sonuç:** AutoGen bir **motor**, bir asistan değil. Tool yazmak kullanan
> tarafın işi. Bu bir eksiklik değil, bir kapsam tercihi — ama kurulumdan sonra
> hazır yetenek bekleyen bir plan buna göre düzeltilmeli.

### Tool nasıl yazılıyor

Bir fonksiyon + docstring yetiyor; şema **imzadan** çıkarılıyor:

```python
def scan_facts(query: str) -> str:
    "Son taramanın özetini döndürür."      # docstring = tarif = arayüz
    ...

FunctionTool(scan_facts, description=scan_facts.__doc__)
```

Modele giden fonksiyon değil, **şeması**:

```json
{"name": "scan_facts",
 "description": "Son taramanın özetini döndürür.",
 "parameters": {"type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]}}
```

**Üç kural:**

1. **Docstring arayüzdür** — modelin o tool'a *ne zaman* uzanacağına karar
   verdiği tek metin. Dokümantasyon değil.
2. **Şemalar her turda ödeniyor.** 17 tool = her istekte 17 şema; `docs/06`
   bunun canlı bir zaman aşımına yol açtığını kaydediyor.
3. **Tip ipucu zorunlu.** `query: str` yoksa şema üretilemiyor.

### Workbench — liste değil, **kaynak**

<div align="center">
<svg viewBox="0 0 600 176" width="600" height="176"><path d="M211,11 L389,11 L389,49 L211,49 Z" fill="#f8f0fc" stroke="none"/><path d="M210.3,9.4 Q300.0,10.0 391.4,10.7 M389.9,11.1 Q389.6,30.0 390.7,50.3 M389.8,51.5 Q300.0,51.4 210.2,49.6 M208.9,49.5 Q210.2,30.0 210.9,9.9" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M211.5,10.1 Q300.0,9.6 389.8,8.7 M390.0,9.3 Q389.8,30.0 391.4,50.4 M391.2,49.7 Q300.0,49.9 211.5,50.6 M209.9,51.5 Q209.8,30.0 210.3,8.4" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="300.0" y="29.2" font-size="8.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Workbench</text><text x="300.0" y="40.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">list_tools / call_tool</text><path d="M21,85 L179,85 L179,127 L21,127 Z" fill="#ebfbee" stroke="none"/><path d="M19.7,83.4 Q100.0,85.3 180.0,83.1 M180.4,83.9 Q179.7,106.0 179.9,127.5 M181.1,126.6 Q100.0,127.3 20.4,129.0 M20.9,129.4 Q20.9,106.0 20.4,85.1" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M19.2,84.7 Q100.0,84.9 180.4,84.1 M181.4,85.0 Q180.5,106.0 180.9,127.2 M179.0,127.6 Q100.0,129.2 19.5,128.8 M21.6,126.6 Q19.8,106.0 19.7,85.2" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="100.0" y="105.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">StaticWorkbench</text><text x="100.0" y="116.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">elindeki fonksiyonlar</text><path d="M221,85 L379,85 L379,127 L221,127 Z" fill="#e7f5ff" stroke="none"/><path d="M220.5,85.1 Q300.0,84.0 380.4,82.9 M381.5,83.7 Q380.8,106.0 380.1,126.8 M379.5,126.9 Q300.0,128.0 220.6,128.6 M221.1,129.5 Q220.2,106.0 221.1,82.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M220.5,85.2 Q300.0,84.1 380.1,83.8 M379.1,85.1 Q380.4,106.0 378.5,128.8 M378.6,127.0 Q300.0,126.3 219.3,129.0 M221.3,128.3 Q220.6,106.0 220.1,84.0" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="300.0" y="105.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">McpWorkbench</text><text x="300.0" y="116.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">stdio ya da HTTP</text><path d="M421,85 L579,85 L579,127 L421,127 Z" fill="#fff4e6" stroke="none"/><path d="M420.7,83.3 Q500.0,84.2 581.3,83.0 M579.9,82.5 Q579.2,106.0 578.7,128.5 M578.7,126.5 Q500.0,126.5 419.5,128.9 M419.7,127.3 Q419.5,106.0 418.4,82.7" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><path d="M420.4,84.7 Q500.0,83.5 580.5,85.3 M580.4,84.7 Q579.8,106.0 579.4,129.6 M581.3,127.9 Q500.0,129.3 419.1,127.5 M418.9,126.6 Q420.1,106.0 419.2,85.5" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><text x="500.0" y="105.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">GatedWorkbench</text><text x="500.0" y="116.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">bizim — kapı</text><path d="M99.5,82.6 Q100.4,68.0 99.1,52.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M99.5,83.5 Q99.9,68.0 98.7,54.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M100.0,54.0 L103.6,60.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M100.0,54.0 L96.2,60.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M298.8,83.4 Q300.2,68.0 301.4,54.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M300.6,81.8 Q300.3,68.0 299.3,53.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M300.0,54.0 L304.3,59.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M300.0,54.0 L296.0,59.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M501.1,82.4 Q499.4,68.0 500.2,54.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M500.2,81.4 Q500.6,68.0 501.5,52.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M500.0,54.0 L504.2,60.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M500.0,54.0 L495.4,59.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="20" y="148" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Tool tek bir arayüz; workbench BİR KOLEKSİYON — durum ve kaynak paylaşan tool'lar, tek tip sonuç.</text><text x="20" y="164" font-size="7.6" fill="#8a5208" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Ajan hangisiyle konuştuğunu bilmiyor. Kapıyı araya koymayı mümkün kılan tek şey bu.</text></svg>
</div>

<sub>▲ Üç kaynak, tek arayüz · düzenlemek için: [`f_workbench_component.excalidraw`](diagrams/wiki/f_workbench_component.excalidraw) → excalidraw.com'a sürükle</sub>


```python
AssistantAgent(tools=[a, b], workbench=wb)
# ValueError: Tools cannot be used with a workbench.
```

İkisi aynı anda olamıyor, çünkü aynı soruyu **farklı zamanda** cevaplıyorlar.
Liste ajan yazılırken donuyor; kaynak her turda sorulabiliyor — MCP sunucusu
tool listesini çalışma zamanında verdiği için tek doğru soyutlama bu.

Kapıyı oraya koymanın sebebi: workbench, yerel bir Python fonksiyonuyla uzak bir
MCP tool'unu **aynı gören tek yer**. Kural, ajan yazılırken **var olmayan**
tool'lar için de geçerli oluyor.

### Model istemcileri

<div align="center">
<svg viewBox="0 0 600 186" width="600" height="186"><path d="M191,9 L409,9 L409,41 L191,41 Z" fill="#f8f0fc" stroke="none"/><path d="M189.0,9.4 Q300.0,8.1 410.4,8.3 M408.7,8.9 Q409.4,25.0 409.1,41.2 M410.8,43.4 Q300.0,42.6 190.7,41.4 M189.3,41.1 Q189.5,25.0 190.0,8.0" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M189.1,9.2 Q300.0,6.6 410.6,6.7 M409.8,7.6 Q409.5,25.0 411.0,42.6 M409.6,42.1 Q300.0,42.2 191.1,42.7 M188.5,41.5 Q189.4,25.0 189.3,8.9" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="300.0" y="24.2" font-size="8.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">ChatCompletionClient</text><text x="300.0" y="35.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">protokol sınıfı</text><path d="M61,63 L539,63 L539,80 L61,80 Z" fill="#e7f5ff" stroke="none"/><path d="M58.8,61.7 Q300.0,62.4 540.2,61.9 M540.6,61.3 Q540.2,71.5 541.6,80.6 M541.0,80.5 Q300.0,81.0 61.3,81.2 M59.2,80.5 Q60.0,71.5 60.1,63.0" fill="none" stroke="#1971c2" stroke-width="1.1" stroke-linecap="round"/><path d="M61.1,63.4 Q300.0,60.8 538.7,62.5 M538.7,63.5 Q540.2,71.5 539.2,80.7 M540.3,82.3 Q300.0,80.3 61.5,80.9 M61.1,79.9 Q59.7,71.5 59.6,61.1" fill="none" stroke="#1971c2" stroke-width="1.1" stroke-linecap="round"/><text x="70" y="75" font-size="6.8" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">OpenAIChatCompletionClient</text><text x="300" y="75" font-size="7.2" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">OpenAI + uyumlu (Gemini…)</text><path d="M299.1,45.0 Q299.8,51.0 300.2,58.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M299.5,43.0 Q299.8,51.0 300.8,58.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M300.0,58.0 L295.7,51.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M300.0,58.0 L303.7,51.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M61,87 L539,87 L539,104 L61,104 Z" fill="#e7f5ff" stroke="none"/><path d="M59.4,84.4 Q300.0,86.6 539.9,85.1 M540.7,86.0 Q539.8,95.5 538.7,105.6 M540.8,105.9 Q300.0,105.1 58.9,104.3 M58.7,103.9 Q60.0,95.5 60.7,84.6" fill="none" stroke="#1971c2" stroke-width="1.1" stroke-linecap="round"/><path d="M58.8,87.3 Q300.0,85.1 540.7,86.8 M539.1,87.0 Q539.8,95.5 539.5,104.6 M539.3,104.6 Q300.0,106.5 61.0,105.0 M60.3,104.4 Q59.8,95.5 61.0,85.5" fill="none" stroke="#1971c2" stroke-width="1.1" stroke-linecap="round"/><text x="70" y="99" font-size="6.8" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">AzureOpenAIChatCompletionClient</text><text x="300" y="99" font-size="7.2" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Azure OpenAI</text><path d="M61,111 L539,111 L539,128 L61,128 Z" fill="#e7f5ff" stroke="none"/><path d="M59.6,110.3 Q300.0,109.0 540.5,109.3 M540.5,109.1 Q540.0,119.5 539.4,128.8 M538.8,128.3 Q300.0,127.9 61.3,129.1 M60.1,130.5 Q59.6,119.5 60.0,110.9" fill="none" stroke="#1971c2" stroke-width="1.1" stroke-linecap="round"/><path d="M58.8,111.2 Q300.0,111.2 538.5,108.7 M539.7,110.8 Q539.9,119.5 539.8,129.2 M540.7,130.2 Q300.0,128.8 61.5,128.7 M58.5,129.7 Q60.4,119.5 60.3,108.6" fill="none" stroke="#1971c2" stroke-width="1.1" stroke-linecap="round"/><text x="70" y="123" font-size="6.8" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">AzureAIChatCompletionClient</text><text x="300" y="123" font-size="7.2" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">GitHub + Azure barındırılan</text><path d="M61,135 L539,135 L539,152 L61,152 Z" fill="#ebfbee" stroke="none"/><path d="M60.7,132.5 Q300.0,133.2 539.7,134.6 M540.1,132.5 Q540.2,143.5 538.8,151.6 M539.7,153.9 Q300.0,153.0 60.7,153.8 M59.9,154.6 Q59.6,143.5 60.1,134.4" fill="none" stroke="#2f9e44" stroke-width="1.1" stroke-linecap="round"/><path d="M58.9,135.6 Q300.0,135.1 539.6,134.0 M538.6,134.8 Q539.6,143.5 538.8,153.9 M540.8,153.0 Q300.0,153.7 60.6,154.3 M60.9,152.1 Q60.2,143.5 60.8,134.5" fill="none" stroke="#2f9e44" stroke-width="1.1" stroke-linecap="round"/><text x="70" y="147" font-size="6.8" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">ReplayChatCompletionClient</text><text x="300" y="147" font-size="7.2" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">deterministik kuru mod</text><text x="60" y="176" font-size="7.6" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Hepsi aynı protokolü uyguluyor — ajan hangisiyle konuştuğunu bilmiyor. Kuru mod istemcisi de bir istemci.</text></svg>
</div>

<sub>▲ İstemci ve model_info · düzenlemek için: [`f_model_clients.excalidraw`](diagrams/wiki/f_model_clients.excalidraw) → excalidraw.com'a sürükle</sub>


**Tuzak:** OpenAI-*uyumlu* bir endpoint kullanıyorsan `model_info` **zorunlu**.
Verilmezse hata net: `model_info is required when model name is not a valid
OpenAI model`. Azure, vLLM, Ollama, OpenRouter — hepsi bu kapsamda.

---

<a id="s11"></a>
## 11 · Kod yürütücüler

<div align="center">
<svg viewBox="0 0 600 150" width="600" height="150"><path d="M17,31 L133,31 L133,73 L17,73 Z" fill="#f8f9fa" stroke="none"/><path d="M14.7,29.7 Q75.0,28.4 132.7,31.3 M134.1,29.5 Q134.8,52.0 135.1,72.9 M133.5,73.2 Q75.0,74.6 14.4,73.8 M16.3,72.6 Q15.3,52.0 15.4,29.8" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><path d="M14.7,28.9 Q75.0,31.5 134.9,28.5 M134.2,29.2 Q133.3,52.0 135.1,74.9 M135.0,73.1 Q75.0,74.6 16.3,74.1 M15.4,73.7 Q16.5,52.0 16.9,30.0" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><text x="75.0" y="51.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">kod bloğu</text><text x="75.0" y="62.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">modelden</text><path d="M136.4,52.5 Q154.0,52.3 173.2,52.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M135.6,50.5 Q154.0,52.3 172.8,51.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M172.0,52.0 L166.2,56.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M172.0,52.0 L166.0,47.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M175,9 L363,9 L363,47 L175,47 Z" fill="#fff5f5" stroke="none"/><path d="M173.7,6.6 Q269.0,7.4 363.9,8.2 M363.0,6.7 Q364.3,28.0 364.4,46.4 M363.1,48.1 Q269.0,48.5 174.7,49.3 M175.6,49.5 Q174.2,28.0 174.4,7.1" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><path d="M173.0,9.6 Q269.0,7.9 363.8,8.8 M362.8,7.0 Q364.7,28.0 363.2,47.5 M363.8,47.7 Q269.0,48.8 172.7,46.6 M172.7,47.8 Q173.8,28.0 172.4,8.1" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><text x="269.0" y="27.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">LocalCommandLine…</text><text x="269.0" y="38.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">host makinede</text><path d="M175,61 L363,61 L363,99 L175,99 Z" fill="#ebfbee" stroke="none"/><path d="M174.6,59.7 Q269.0,58.5 365.6,59.2 M364.8,59.5 Q364.0,80.0 363.5,100.4 M365.2,99.7 Q269.0,98.3 174.7,100.6 M173.2,98.7 Q173.8,80.0 175.2,58.6" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M174.1,58.9 Q269.0,60.1 364.5,61.3 M363.6,58.8 Q363.6,80.0 365.5,100.1 M365.0,99.6 Q269.0,99.5 172.6,98.8 M173.5,101.2 Q174.3,80.0 174.8,61.5" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="269.0" y="79.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">DockerCommandLine…</text><text x="269.0" y="90.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">konteynerde</text><path d="M365.4,28.9 Q384.0,28.3 400.5,28.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M366.0,26.8 Q384.0,28.7 402.0,28.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M402.0,28.0 L395.6,32.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M402.0,28.0 L395.6,23.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M365.1,79.4 Q384.0,80.0 401.7,81.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M365.0,79.0 Q384.0,79.3 403.6,81.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M402.0,80.0 L395.3,84.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M402.0,80.0 L395.2,75.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M405,9 L587,9 L587,47 L405,47 Z" fill="#fff5f5" stroke="none"/><path d="M404.1,9.4 Q496.0,7.4 589.0,8.1 M587.9,8.7 Q587.5,28.0 588.9,47.2 M588.2,48.0 Q496.0,46.8 403.7,47.1 M402.8,48.4 Q404.0,28.0 405.3,8.9" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><path d="M404.0,8.6 Q496.0,8.4 586.8,6.6 M587.6,8.5 Q587.4,28.0 587.7,47.5 M587.5,48.8 Q496.0,47.0 405.0,47.0 M402.6,47.7 Q404.8,28.0 404.0,9.1" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><text x="496.0" y="31.2" font-size="7.8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">→ host'un her şeyi</text><path d="M405,61 L587,61 L587,99 L405,99 Z" fill="#ebfbee" stroke="none"/><path d="M404.2,59.6 Q496.0,58.8 589.3,58.8 M589.0,59.4 Q588.4,80.0 589.4,98.6 M586.7,100.2 Q496.0,100.4 405.5,99.4 M403.3,99.8 Q404.2,80.0 404.2,59.2" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M404.0,61.3 Q496.0,60.7 587.3,61.5 M586.5,60.5 Q588.5,80.0 588.7,99.9 M589.6,98.9 Q496.0,101.6 405.2,101.3 M405.2,99.9 Q403.6,80.0 405.3,61.2" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="496.0" y="83.2" font-size="7.8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">→ yalıtılmış</text><text x="16" y="122" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Her kod bloğu bir dosyaya yazılıp AYRI BİR SÜREÇTE koşuyor — yani bloklar arası değişken paylaşımı yok.</text><text x="16" y="138" font-size="7.6" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Local, modelin yazdığı kodu makinende koşturur. Bu bir tercih değil, bir güven kararıdır.</text></svg>
</div>

<sub>▲ Yerel · Docker · Jupyter · düzenlemek için: [`f_code_executors.excalidraw`](diagrams/wiki/f_code_executors.excalidraw) → excalidraw.com'a sürükle</sub>


Resmî sekiz desenin sonuncusu **Code Execution**, ve diğer yedisinden farkı şu:
onlar orkestrasyon deseni, bu bir **yetenek**. Modelin yazdığı Python'u
çalıştırıyor.

### Dört yürütücü

`autogen_ext.code_executors` altında **[ölçüldü]**:

| Yürütücü | İzolasyon | Not |
|---|---|---|
| `local` | **yok** | Kod doğrudan sunucu sürecinin yanında koşuyor |
| `docker` | konteyner | Kılavuzun önerdiği |
| `jupyter` | çekirdek | Ekstra gerekiyor · **durum taşıyor** |
| `docker_jupyter` | konteyner + çekirdek | İkisinin birleşimi |
| `azure` | uzak | Azure Container Apps |

Kılavuz yerel yürütücü için açık uyarı veriyor: **modelin ürettiği kodu izolesiz
çalıştırmak risklidir.**

### Docker yürütücünün parametreleri — ve orada olmayanlar

`DockerCommandLineCodeExecutor` **[ölçüldü]**:

```python
DockerCommandLineCodeExecutor(
    image="python:3-slim",      # varsayılan
    timeout=60,                  # saniye
    work_dir=None,               # host'ta bağlanan dizin
    auto_remove=True,            # konteyner çıkışta siliniyor
    stop_container=True,
    extra_volumes=None,
    device_requests=None,        # GPU
    init_command=None,           # konteyner açılışında koşacak komut
)
```

**Ve listede olmayanlar, listede olanlardan daha önemli:**

| Yok | Sonucu |
|---|---|
| `network_mode` | Konteyner varsayılan **bridge** ağında — **interneti var** |
| `user` | İçeride **root** |
| `read_only` | Kök dosya sistemi **yazılabilir** |
| `mem_limit` · `nano_cpus` · `pids_limit` | **Kaynak sınırı yok** |
| `cap_drop` | Hiçbir yetki düşürülmüyor |

Bu bir yapılandırma eksikliği değil, **API'de o parametreler yok** — kaynağında
ağ ile ilgili tek kelime geçmiyor.

> **Sonuç:** *"kod sandbox'ta koşuyor"* cümlesi bu yürütücüyle kurulamaz.
> Kurulabilecek cümle: *"kod izole bir konteynerde koşuyor, ve konteynerin ağ
> erişimi var."*

Sertleştirme mümkün ama bedava değil: `start()` override edilip
`containers.create(..., network_mode="none", user="1000", mem_limit="512m")`
geçilebilir — bu **yukarı akışın iç koduna bağımlılık** yaratıyor ve sürüm
değişince sessizce kırılıyor. Bakım modundaki bir projede risk daha yüksek.

### Konteynerin ömrü: çağrı başına mı, süreç başına mı

Konteyner ayağa kaldırmak **2–3 saniye**, ve bu süre kullanıcının beklediği
zamana ekleniyor. İki seçenek:

| | Çağrı başına | Süreç başına |
|---|---|---|
| Gecikme | her çağrıda 2–3 sn | bir kez, açılışta |
| Turlar arası durum | temiz | **taşınıyor** |
| İzolasyon | konteyner ↔ host, tur ↔ tur | yalnız konteyner ↔ host |

`start()` / `stop()` sunucunun yaşam döngüsüne bağlanırsa süreç başına tek
konteyner olur — hızlı, ama bir turun `/tmp`'ye yazdığını sonraki tur görüyor.

### Tool'a dönüşmesi — ve tarifin önemi

`PythonCodeExecutionTool(executor)` yürütücüyü normal bir tool'a çeviriyor, yani
**aynı döngüden, aynı workbench'ten, aynı kapıdan** geçiyor. Ayrı bir yol yok.

Ama varsayılan tarifi tek cümle: **`"Execute Python code blocks."`** Bu tarifle
model kodu bir *kaçış kapağı* değil, bir *genel çözüm* sanıyor ve her hesabı
yeniden icat ediyor — mevcut tool'lar boşta kalıyor.

Tarif, modelin bu tool'a **ne zaman** uzanacağına karar verdiği tek metin. Rolü
anlatan bir tarif şunu söylemeli: *"önce mevcut tool'lara bak; sorulanı
karşılayan yoksa kod yaz."*

### Kapı için özel bir kanca gerekiyor

Ad bazlı bir kapı **bu tool'u göremiyor.** `"CodeExecutor"` tipik dışarı-yazma
işaretlerinin (`send`, `post`, `write`, `delete`) hiçbirine uymuyor, yani ada
bakan bir filtre onu sessizce geçiriyor.

Çözüm: `before_tool_call` seviyesinde **ada değil türe** bakan bir kanca, ve
onayı `(tool, argümanlar)` imzasına bağlamak — böylece kod değişirse eski onay
tutmuyor.

> **Ve onay tüketildikten sonra:** aynı soruyu modele tekrar sormak **farklı bir
> program** üretiyor **[ölçüldü]**. Onaylananla çalışanın aynı olmasının tek
> yolu, çalıştırılacak olanın **onaylanan metin** olması — yeniden üretilen değil.

### MAF tarafı

MAF'ta karşılığı **hosted tool** olarak geliyor: `SupportsCodeInterpreterTool`
sözleşmesini karşılayan bir istemci, kodu **sağlayıcı tarafında** çalıştırıyor.
Ayrıca `MontyCodeActProvider` ile sandbox'lı, çapraz platform bir yorumlayıcı
seçeneği var **[teyitsiz]**.

Fark: AutoGen'de konteyner **senin makinende**, MAF'ın hosted yolunda
**sağlayıcıda**. İkisi farklı güven kararı — birinde altyapı senin, diğerinde
veri dışarı çıkıyor.

---

<a id="s12"></a>
## 12 · Ölçülmüş tuzaklar

<div align="center">
<svg viewBox="0 0 600 200" width="600" height="200"><path d="M17,11 L583,11 L583,39 L17,39 Z" fill="#fff5f5" stroke="none"/><path d="M17.1,8.7 Q300.0,10.0 583.6,8.6 M583.0,9.4 Q583.8,25.0 585.0,38.6 M583.4,41.0 Q300.0,41.2 15.9,40.4 M15.1,39.8 Q16.5,25.0 14.9,10.9" fill="none" stroke="#c92a2a" stroke-width="1.3" stroke-linecap="round"/><path d="M16.8,9.5 Q300.0,8.3 584.2,9.0 M583.4,9.9 Q583.5,25.0 585.4,40.3 M583.1,41.3 Q300.0,40.6 16.7,40.4 M17.3,38.8 Q16.1,25.0 15.6,11.0" fill="none" stroke="#c92a2a" stroke-width="1.3" stroke-linecap="round"/><text x="28" y="23" font-size="8.2" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">max_tool_iterations</text><text x="28" y="35" font-size="6.8" fill="#c92a2a" font-family="DejaVu Sans Mono, monospace">varsayılan: 1</text><text x="210" y="29" font-size="8.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">model tool sonucunu GÖRMEDEN cevaplar</text><path d="M17,49 L583,49 L583,77 L17,77 Z" fill="#fff5f5" stroke="none"/><path d="M16.4,48.5 Q300.0,48.6 583.9,48.2 M585.4,47.3 Q584.1,63.0 583.2,77.4 M584.4,78.3 Q300.0,79.2 14.6,77.2 M15.3,78.5 Q16.5,63.0 15.7,46.9" fill="none" stroke="#c92a2a" stroke-width="1.3" stroke-linecap="round"/><path d="M16.4,47.2 Q300.0,49.7 584.3,47.4 M582.5,47.8 Q583.6,63.0 584.4,78.8 M582.6,77.9 Q300.0,79.4 16.3,77.9 M15.3,78.7 Q15.5,63.0 16.1,49.4" fill="none" stroke="#c92a2a" stroke-width="1.3" stroke-linecap="round"/><text x="28" y="61" font-size="8.2" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">model_context</text><text x="28" y="73" font-size="6.8" fill="#c92a2a" font-family="DejaVu Sans Mono, monospace">varsayılan: yok</text><text x="210" y="67" font-size="8.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">ajanın belleği yoktur</text><path d="M17,87 L583,87 L583,115 L17,115 Z" fill="#fff5f5" stroke="none"/><path d="M16.4,86.3 Q300.0,84.4 583.8,84.6 M583.4,84.9 Q584.1,101.0 585.1,114.7 M585.3,117.4 Q300.0,117.7 16.5,116.0 M14.6,115.2 Q16.3,101.0 16.0,84.8" fill="none" stroke="#c92a2a" stroke-width="1.3" stroke-linecap="round"/><path d="M16.3,85.2 Q300.0,87.4 584.3,87.1 M585.2,86.9 Q583.8,101.0 584.2,117.3 M583.3,115.1 Q300.0,114.6 14.9,116.7 M16.1,117.6 Q16.4,101.0 15.5,85.1" fill="none" stroke="#c92a2a" stroke-width="1.3" stroke-linecap="round"/><text x="28" y="99" font-size="8.2" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">model_client_stream</text><text x="28" y="111" font-size="6.8" fill="#c92a2a" font-family="DejaVu Sans Mono, monospace">varsayılan: False</text><text x="210" y="105" font-size="8.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">token akışı hiç yayılmaz</text><path d="M17,125 L583,125 L583,153 L17,153 Z" fill="#fff5f5" stroke="none"/><path d="M16.6,123.8 Q300.0,123.5 585.4,123.6 M584.7,124.1 Q583.6,139.0 585.5,154.6 M585.6,154.7 Q300.0,152.6 15.8,153.3 M15.5,155.4 Q16.5,139.0 15.3,123.1" fill="none" stroke="#c92a2a" stroke-width="1.3" stroke-linecap="round"/><path d="M16.5,122.9 Q300.0,125.6 583.9,123.2 M584.5,124.6 Q584.1,139.0 585.3,155.6 M583.4,155.6 Q300.0,154.7 16.1,154.8 M15.0,153.3 Q15.6,139.0 15.7,123.8" fill="none" stroke="#c92a2a" stroke-width="1.3" stroke-linecap="round"/><text x="28" y="137" font-size="8.2" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">sonlandırma</text><text x="28" y="149" font-size="6.8" fill="#c92a2a" font-family="DejaVu Sans Mono, monospace">varsayılan: yok</text><text x="210" y="143" font-size="8.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">takım tavansız koşar</text><text x="16" y="176" font-size="9" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Dördünün ortak yanı: HİÇBİRİ HATA VERMEZ.</text><text x="16" y="190" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Sistem çalışır, sonuç yanlış olur — ve aramak için önce yanlış olduğunu bilmen gerekir.</text></svg>
</div>

<sub>▲ Hiçbiri istisna fırlatmıyor · düzenlemek için: [`f_gotchas.excalidraw`](diagrams/wiki/f_gotchas.excalidraw) → excalidraw.com'a sürükle</sub>


| Tuzak | Sonuç |
|---|---|
| `tools=` ve `workbench=` birlikte | `ValueError` — tek net hata |
| `model_context` verilmemiş | Ajanın **belleği yok**, hata da vermiyor |
| OpenAI-*uyumlu* endpoint | `model_info` **zorunlu** |
| `max_tool_iterations` = 1 | Zincirleme davranış sessizce imkânsız |
| Dış runtime, ajan çöküyor | **Fırlatmaz, asar** |
| `description` boş ajan | `SelectorGroupChat` **kör** seçiyor |
| `Handoff` adı küçük harfe düşüyor | Elle yazınca eşleşmiyor |
| `stop_when_idle()` | Handler çökerse bariyer erken açılıyor |

> **Ortak nokta:** bulunan hataların **hiçbiri istisna fırlatmadı.** Sıfır
> döndü, boş kaldı, asılı kaldı, ya da hata metnini cevap diye sundu.
> Core'u öğrenmenin yolu API'sini okumak değil, **arıza davranışını ölçmek**.

---

<a id="s13"></a>
## 13 · MAF ne getirdi

<div align="center">
<svg viewBox="0 0 600 156" width="600" height="156"><path d="M13,9 L391,9 L391,27 L13,27 Z" fill="#e7f5ff" stroke="none"/><path d="M13.2,8.7 Q202.0,6.3 390.5,8.2 M393.2,9.1 Q392.2,18.0 391.4,28.3 M391.3,27.8 Q202.0,28.0 11.7,27.9 M12.7,26.5 Q12.4,18.0 11.5,9.5" fill="none" stroke="#1971c2" stroke-width="1.2" stroke-linecap="round"/><path d="M12.3,8.5 Q202.0,8.9 392.6,6.9 M391.3,7.0 Q392.1,18.0 392.6,26.5 M392.7,29.3 Q202.0,29.2 11.4,29.1 M10.9,29.3 Q12.3,18.0 10.9,7.9" fill="none" stroke="#1971c2" stroke-width="1.2" stroke-linecap="round"/><text x="22" y="22" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Model Clients</text><text x="160" y="22" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:1984</text><text x="216" y="22" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">sağlayıcıya konuşan şey</text><path d="M13,33 L391,33 L391,51 L13,51 Z" fill="#f8f0fc" stroke="none"/><path d="M10.7,32.5 Q202.0,30.5 393.1,30.8 M392.4,32.3 Q392.1,42.0 390.8,52.6 M393.4,52.5 Q202.0,53.4 12.5,53.4 M11.0,52.4 Q11.8,42.0 13.3,33.4" fill="none" stroke="#5f3dc4" stroke-width="1.2" stroke-linecap="round"/><path d="M12.3,31.6 Q202.0,33.1 392.9,32.4 M392.8,32.0 Q391.7,42.0 392.5,51.1 M392.8,52.7 Q202.0,52.8 10.5,51.1 M12.2,51.5 Q11.9,42.0 11.6,33.4" fill="none" stroke="#5f3dc4" stroke-width="1.2" stroke-linecap="round"/><text x="22" y="46" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Model Context</text><text x="160" y="46" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:2341</text><text x="216" y="46" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">modele NE gideceğine karar veren şey</text><path d="M13,57 L391,57 L391,75 L13,75 Z" fill="#ebfbee" stroke="none"/><path d="M11.8,57.4 Q202.0,55.6 391.4,56.1 M393.1,57.0 Q392.3,66.0 391.2,76.4 M393.1,76.0 Q202.0,76.9 11.3,77.5 M12.2,77.5 Q12.0,66.0 11.3,55.9" fill="none" stroke="#2f9e44" stroke-width="1.2" stroke-linecap="round"/><path d="M11.5,57.1 Q202.0,57.5 392.4,57.5 M393.2,56.0 Q391.9,66.0 391.1,75.5 M392.4,75.9 Q202.0,76.0 11.6,75.3 M12.6,76.4 Q12.0,66.0 10.5,55.2" fill="none" stroke="#2f9e44" stroke-width="1.2" stroke-linecap="round"/><text x="22" y="70" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Tools</text><text x="160" y="70" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:2473</text><text x="216" y="70" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">modelin çağırabildiği fonksiyonlar</text><path d="M13,81 L391,81 L391,99 L13,99 Z" fill="#fff4e6" stroke="none"/><path d="M12.0,79.3 Q202.0,80.6 392.3,78.5 M393.4,79.5 Q392.0,90.0 390.9,100.6 M391.5,99.2 Q202.0,99.6 10.4,98.7 M11.7,101.1 Q12.2,90.0 13.5,80.8" fill="none" stroke="#e8590c" stroke-width="1.2" stroke-linecap="round"/><path d="M10.8,81.3 Q202.0,79.5 390.9,78.9 M391.3,79.3 Q392.1,90.0 391.0,98.6 M393.4,98.7 Q202.0,101.7 11.8,100.5 M13.2,101.5 Q12.1,90.0 12.3,79.8" fill="none" stroke="#e8590c" stroke-width="1.2" stroke-linecap="round"/><text x="22" y="94" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Workbench (+ MCP)</text><text x="160" y="94" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:2841</text><text x="216" y="94" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">tool'ların toplandığı arayüz</text><path d="M13,105 L391,105 L391,123 L13,123 Z" fill="#fff5f5" stroke="none"/><path d="M11.5,103.0 Q202.0,104.5 391.9,104.0 M391.9,103.6 Q392.2,114.0 390.7,125.5 M392.2,122.7 Q202.0,123.9 11.4,122.5 M12.5,124.0 Q12.3,114.0 12.6,103.8" fill="none" stroke="#c92a2a" stroke-width="1.2" stroke-linecap="round"/><path d="M13.6,102.8 Q202.0,104.1 392.6,103.7 M390.9,104.3 Q392.2,114.0 390.7,125.0 M391.4,123.8 Q202.0,124.3 11.7,123.8 M11.7,123.2 Q11.8,114.0 12.8,104.3" fill="none" stroke="#c92a2a" stroke-width="1.2" stroke-linecap="round"/><text x="22" y="118" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold">Code Executors</text><text x="160" y="118" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace">05:3054</text><text x="216" y="118" font-size="7.4" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">kodu nerede koşturacağın</text><path d="M413,9 L587,9 L587,123 L413,123 Z" fill="#f8f9fa" stroke="none"/><path d="M411.5,7.5 Q500.0,7.5 589.5,8.5 M589.3,8.7 Q586.4,66.0 588.4,123.4 M586.8,123.9 Q500.0,123.6 412.6,124.0 M412.5,124.3 Q412.0,66.0 412.2,8.2" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><path d="M412.2,6.4 Q500.0,7.6 588.4,7.1 M586.6,8.1 Q587.4,66.0 587.4,123.9 M588.7,125.1 Q500.0,122.8 412.2,125.2 M413.2,123.1 Q413.8,66.0 410.8,7.5" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><text x="500.0" y="23" font-size="8.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Component config</text><text x="500" y="30" font-size="6.6" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">05:1888</text><text x="424" y="56" font-size="6.6" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">dump_component / load_component</text><text x="424" y="70" font-size="7" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">her bileşen JSON'a yazılıp</text><text x="424" y="82" font-size="7" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">geri yüklenebiliyor</text><text x="424" y="100" font-size="7" fill="#2f9e44" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">→ yapılandırma kod değil VERİ</text><text x="12" y="146" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Beşi de değiştirilebilir yüzey. Bir ajan bunların hangi uygulamasıyla konuştuğunu bilmiyor — kapıyı kurmayı mümkün kılan da bu.</text></svg>
</div>

<sub>▲ MAF'ın eklediği katmanlar · düzenlemek için: [`f_components.excalidraw`](diagrams/wiki/f_components.excalidraw) → excalidraw.com'a sürükle</sub>


AutoGen'de **karşılığı olmayan** beş şey:

| Yetenek | Ne yapıyor |
|---|---|
| **Middleware** | Ajan / sohbet / fonksiyon seviyelerinde ara katman; her biri turu **durdurabiliyor** |
| **Checkpoint** | Workflow durumu diske yazılıp geri yükleniyor |
| **İnsan döngüde** | `ctx.request_info()` + `@response_handler` — çerçevenin **içinde** |
| **Harness** | `create_harness_agent` — todo, plan/execute kipleri, dosya belleği, onay, OTel |
| **FIDES** | Bütünlük + gizlilik etiketleri; politika hassas tool çalışmadan **önce** zorlanıyor |

Kılavuzun kendi cümlesi **[kaynak]**:

> *"AutoGen's `Team` abstraction runs continuously once started and doesn't
> provide built-in mechanisms to pause execution for human input."*

### Mimari fark: kontrol akışından veri akışına

* **GraphFlow** — *control-flow*: kenarlar geçiş, mesajlar **herkese** yayınlanır
* **Workflow** — *data-flow*: mesajlar **belirli kenarlardan**, yürütücü girdisi
  hazır olunca tetikleniyor

Bu, §5'teki sessiz kardeş kaybının kökeni.

---

<a id="s14"></a>
## 14 · MAF ne kaybettirdi

| Yetenek | AutoGen | MAF |
|---|---|---|
| Dağıtık runtime (gRPC) | var (deneysel) | **yok** — "planned" |
| Model yanıtı önbelleği | `ChatCompletionCache` | **yok** — "🚧 Planned" |
| Aktör modeli / topic | `autogen-core` | **yok** |

### Ve hızın faturası

* **1.0 GA'dan sonra iki ayda 15 kırıcı değişiklik** — Microsoft'un kendi
  🔴 işaretlemesiyle **[kaynak]**
* 36 paketin **8'i** kararlı; 22 `beta`, 6 `alpha`
* Harness, FIDES, beceriler → hepsi `experimental` ve gerçekten
  `ExperimentalWarning` fırlatıyor **[ölçüldü]**

### Sürüm hızı — ölçüldü

| Paket | Son sürüm | Kaç gün önce |
|---|---|---:|
| **autogen-agentchat** | 0.7.5 | **323** |
| agent-framework | 1.14.0 | 5 |
| langgraph | 1.2.11 | 8 |
| openai-agents | 0.22.0 | **0** |

---

<a id="s15"></a>
## 15 · Geçiş haritası

Microsoft'un kendi göç kılavuzundan **[kaynak]**:

| AutoGen | MAF |
|---|---|
| `AssistantAgent(model_client=…)` | `Agent(client=…)` |
| `FunctionTool(fn)` | `@tool` — şemayı imzadan çıkarıyor |
| `RoundRobinGroupChat` | `SequentialBuilder` |
| `SelectorGroupChat` | `GroupChatBuilder(selection_func=…)` |
| `Swarm` | `HandoffBuilder` |
| `MagenticOneGroupChat` | `MagenticBuilder` |
| `GraphFlow` | `WorkflowBuilder` |
| `model_context` (ajana ait) | `AgentSession` (çağrıya ait) |

> **Dikkat:** kılavuz Swarm ve Selector'ı *"currently in development"* diyor,
> ama ikisi de 1.14.0'da **var ve `released`** **[ölçüldü]**. Kılavuz kendi
> paketinin gerisinde — paketi aç, `dir()` çek.

---

<sub>Üretim: `python docs/tools/make_wiki.py` · şemalar `docs/diagrams/figures.py`
· kaynak metinler `docs/05` · `docs/08` · `docs/20` · `docs/21` · Türkçe rehberler
`docs/11` · `docs/22`</sub>
