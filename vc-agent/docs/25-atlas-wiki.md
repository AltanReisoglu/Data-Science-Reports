# Atlas — ajan altyapısı wiki'si

> **Bu ne:** KKB'de bir ajan sistemi kurarken bilinmesi gerekenler. Tek dosya,
> arayarak okunmak için. `Ctrl+F` ile gel, cevabı al, kapat.
>
> **Kaynak:** `vc-agent` deposu · 484 test · her sayı ölçüldü.
> Etiketler: **[ölçüldü]** koşturuldu · **[kaynak]** birincil metinden ·
> **[teyitsiz]** okundu, koşturulmadı.
>
> **Şemalar:** okumak için gömülü, değiştirmek için her birinin altında
> `.excalidraw` bağı var — dosyayı [excalidraw.com](https://excalidraw.com)'a
> sürüklemek yetiyor.

---

## İçindekiler

1. [Sözlük — beş terim](#s1)
2. [AutoGen: üç katman](#s2)
3. [Aktör modeli: ajanlar nasıl konuşuyor](#s3)
4. [Tool döngüsü ve sessiz varsayılanlar](#s4)
5. [Workbench: tool'ların tek kapısı](#s5)
6. [Onay kapısı](#s6)
7. [Takımlar ve faturaları](#s7)
8. [Kod yürütme ve Docker](#s8)
9. [Zamanlayıcı](#s9)
10. [OpenClaw'dan alınanlar](#s10)
11. [Denetim: iki kayıt hattı](#s11)
12. [Çerçeve seçimi](#s12)
13. [Bilinen sınırlar](#s13)

---

<a id="s1"></a>
## 1 · Sözlük

Beş terim; wiki'nin geri kalanı bunları kullanıyor.

| Terim | Ne demek |
|---|---|
| **Ajan** | Bir model + talimat + tool listesi + hafıza. Nesne olarak bir Python sınıfı. |
| **Tool** | Ajanın çağırabildiği fonksiyon. Model fonksiyonu görmüyor, **tarifini** görüyor. |
| **Runtime** | Ajanlar arası mesajı taşıyan postane. Ajan ajanı çağırmıyor; runtime'a mesaj veriyor. |
| **Workbench** | Tool listesi değil, tool **kaynağı**. "Elimde ne var" diye her turda sorulabiliyor. |
| **Harness** | Dil modelini iş yapabilen bir ajana çeviren runtime iskelesi — oturum, onay, bellek, zamanlama. |

---

<a id="s2"></a>
## 2 · AutoGen: üç katman

<div align="center">
<svg viewBox="0 0 600 182" width="600" height="182"><path d="M21,13 L579,13 L579,47 L21,47 Z" fill="#f8f9fa" stroke="none"/><path d="M19.9,12.2 Q300.0,12.0 579.6,13.1 M579.8,12.8 Q580.5,30.0 580.3,49.3 M579.5,47.4 Q300.0,48.8 18.8,46.6 M19.4,48.3 Q20.0,30.0 21.5,13.6" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><path d="M19.5,11.6 Q300.0,13.0 581.3,11.1 M579.2,11.8 Q579.5,30.0 580.2,49.5 M581.1,47.5 Q300.0,47.2 19.5,49.2 M20.7,46.7 Q20.1,30.0 18.6,11.8" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><text x="34" y="28" font-size="9.4" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">autogen_ext</text><text x="34" y="42" font-size="7.6" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">dış dünya</text><text x="210" y="36" font-size="8" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">model istemcileri · MCP · kod yürütücüler</text><path d="M21,57 L579,57 L579,91 L21,91 Z" fill="#e7f5ff" stroke="none"/><path d="M19.3,57.0 Q300.0,55.3 581.3,55.1 M579.0,55.0 Q580.6,74.0 580.7,91.6 M578.5,92.1 Q300.0,90.3 18.8,93.1 M19.2,92.3 Q19.8,74.0 21.2,54.4" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M21.3,55.7 Q300.0,55.0 578.7,57.4 M579.8,55.0 Q579.4,74.0 578.5,92.8 M580.8,92.6 Q300.0,91.0 19.5,91.3 M20.9,91.9 Q20.1,74.0 20.0,55.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="34" y="72" font-size="9.4" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">autogen_agentchat</text><text x="34" y="86" font-size="7.6" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">günlük iş</text><text x="210" y="80" font-size="8" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">AssistantAgent · 5 takım · 11 sonlandırma</text><path d="M21,101 L579,101 L579,135 L21,135 Z" fill="#f8f0fc" stroke="none"/><path d="M21.5,99.3 Q300.0,101.2 581.5,101.0 M579.2,100.6 Q580.1,118.0 580.1,136.3 M581.5,136.7 Q300.0,134.2 20.1,134.9 M21.0,135.2 Q19.4,118.0 20.6,99.4" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M20.6,98.9 Q300.0,99.0 578.6,101.6 M579.6,100.9 Q580.1,118.0 579.1,136.9 M580.7,136.8 Q300.0,137.0 18.7,135.6 M20.0,135.8 Q20.5,118.0 18.6,98.9" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="34" y="116" font-size="9.4" fill="#1e1e1e" font-family="DejaVu Sans Mono, monospace">autogen_core</text><text x="34" y="130" font-size="7.6" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">aktör modeli</text><text x="210" y="124" font-size="8" fill="#454c53" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">AgentId(type,key) · runtime · topic · abonelik</text><text x="20" y="156" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Ayıran şey en alt katman: ajanlar gerçekten aktör — kendi mailbox'ı olan, mesajı tipe göre yönlendiren birimler.</text><text x="20" y="170" font-size="7.6" fill="#2f9e44" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Yukarıdan başla; aşağı inmek her zaman mümkün.</text></svg>
</div>

<sub>▲ AutoGen'in üç katmanı · düzenlemek için: [`f_layers.excalidraw`](diagrams/wiki/f_layers.excalidraw) → excalidraw.com'a sürükle</sub>


* **`autogen_core`** — aktör modeli. Kimlik, runtime, topic, abonelik.
* **`autogen_agentchat`** — günlük iş. Hazır ajan, beş takım tipi, on bir sonlandırma koşulu.
* **`autogen_ext`** — dış dünya. Model istemcileri, MCP, kod yürütücüler.

**Kural:** yukarıdan başla. AgentChat'in çözdüğü bir problemi core'da yeniden
çözmek, aynı işi daha az testle yapmak demek. Aşağı inmek zorunda değilsin ama
**inebildiğini bilmek** bir güvence — bu projede paralel dal kaybını AgentChat'te
çözemedik, core'a inip çözdük.

---

<a id="s3"></a>
## 3 · Aktör modeli

<div align="center">
<svg viewBox="0 0 600 180" width="600" height="180"><path d="M5.0,4.6 Q300.0,7.4 594.2,7.6 M593.5,5.3 Q594.1,90.0 593.3,173.3 M592.6,173.8 Q300.0,175.5 6.0,173.1 M6.1,173.3 Q7.3,90.0 5.8,5.3" fill="none" stroke="#868e96" stroke-width="1.2" stroke-linecap="round" stroke-dasharray="7 5"/><path d="M7.5,4.8 Q300.0,5.1 594.7,6.6 M593.6,7.1 Q594.8,90.0 592.9,173.0 M595.2,173.7 Q300.0,175.7 4.9,174.2 M6.9,172.7 Q5.8,90.0 6.3,7.3" fill="none" stroke="#868e96" stroke-width="1.2" stroke-linecap="round" stroke-dasharray="7 5"/><text x="13" y="19" font-size="7.2" fill="#868e96" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">SingleThreadedAgentRuntime</text><path d="M31,47 L147,47 L147,99 L31,99 Z" fill="#ffffff" stroke="none"/><path d="M30.2,47.4 Q89.0,45.8 147.9,46.0 M147.0,46.0 Q147.8,73.0 148.4,100.9 M147.4,98.7 Q89.0,101.5 31.0,100.6 M31.5,101.5 Q29.0,73.0 30.5,46.4" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><path d="M28.4,46.1 Q89.0,44.8 146.6,45.0 M146.5,45.9 Q148.6,73.0 147.8,101.1 M148.4,100.0 Q89.0,99.9 30.5,99.9 M31.6,101.6 Q29.5,73.0 31.1,46.7" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><text x="89.0" y="72.2" font-size="9.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Agent A</text><text x="89.0" y="83.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">type + key</text><path d="M241,35 L357,35 L357,73 L241,73 Z" fill="#e7f5ff" stroke="none"/><path d="M239.1,33.3 Q299.0,33.3 356.6,34.9 M359.1,33.6 Q358.2,54.0 359.5,75.1 M357.1,75.3 Q299.0,75.8 239.9,75.5 M238.6,74.4 Q239.8,54.0 240.9,33.3" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M239.5,35.5 Q299.0,32.5 358.8,32.8 M356.7,32.6 Q358.4,54.0 359.0,73.0 M357.8,73.0 Q299.0,73.8 240.7,72.8 M238.8,73.7 Q240.2,54.0 239.1,33.3" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="299.0" y="57.2" font-size="9.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Agent B</text><path d="M241,97 L357,97 L357,135 L241,135 Z" fill="#e7f5ff" stroke="none"/><path d="M241.0,95.4 Q299.0,97.7 359.2,95.1 M359.1,96.5 Q358.2,116.0 356.7,137.6 M357.2,136.9 Q299.0,137.0 239.5,135.3 M238.7,136.3 Q239.3,116.0 239.2,96.3" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M239.9,97.5 Q299.0,95.5 357.9,96.2 M357.0,94.9 Q357.4,116.0 359.3,137.0 M357.0,136.8 Q299.0,136.9 241.4,135.0 M241.2,136.3 Q240.7,116.0 239.7,94.7" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="299.0" y="119.2" font-size="9.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Agent C</text><path d="M453,47 L569,47 L569,99 L453,99 Z" fill="#ebfbee" stroke="none"/><path d="M453.5,45.2 Q511.0,44.3 570.7,45.2 M570.3,45.3 Q569.3,73.0 569.0,100.7 M569.1,100.2 Q511.0,101.6 453.1,100.4 M453.3,99.1 Q451.5,73.0 450.5,45.3" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M450.6,45.0 Q511.0,45.8 569.6,46.2 M569.6,47.3 Q570.8,73.0 571.5,100.5 M570.3,98.8 Q511.0,99.3 450.5,98.5 M452.6,101.5 Q452.9,73.0 450.5,46.4" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="511.0" y="72.2" font-size="9.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">ClosureAgent</text><text x="511.0" y="83.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">toplayıcı</text><path d="M150.7,65.4 Q194.0,59.9 239.6,52.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M150.8,67.3 Q194.0,60.2 238.8,54.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M238.0,54.0 L232.8,59.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M238.0,54.0 L231.1,51.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="194.0" y="54.0" font-size="7.4" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" text-anchor="middle">publish</text><path d="M151.2,81.7 Q193.4,98.7 238.9,113.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M150.4,81.6 Q193.9,97.3 238.3,112.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M238.0,112.0 L230.2,114.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M238.0,112.0 L234.1,106.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M359.6,54.8 Q404.9,61.0 450.3,65.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M358.7,54.8 Q404.8,61.5 448.5,66.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M450.0,66.0 L443.2,68.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M450.0,66.0 L444.7,61.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M361.3,115.2 Q405.2,99.5 448.4,81.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M359.0,114.9 Q405.3,99.7 451.3,82.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M450.0,82.0 L445.6,88.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M450.0,82.0 L442.5,80.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="30" y="122" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">mesaj kuyruğu · tek iş parçacığı · sıra korunur</text><text x="30" y="136" font-size="7.6" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">ajanlar birbirini çağırmaz — runtime taşır</text></svg>
</div>

<sub>▲ Ajan ajanı çağırmıyor — runtime'a mesaj veriyor · düzenlemek için: [`f_actor.excalidraw`](diagrams/wiki/f_actor.excalidraw) → excalidraw.com'a sürükle</sub>


Bir ajan başka bir ajanın nesnesini elinde tutmuyor. Runtime'a mesaj veriyor,
teslimatı runtime yapıyor. Bunun bedeli var — araya bir katman giriyor ve
*"kim kimi çağırdı"* sorusunun cevabı yığın izinde görünmüyor. Karşılığında
üç şey kazanıyorsun: yeni ajan eklemek çağıran kodu **değiştirmiyor**, bütün
mesajlar tek noktadan geçtiği için müdahale ve ölçüm oraya takılıyor, ve aynı
sınıftan istediğin kadar örnek bedava.

### İki iletişim biçimi — fark adresleme değil, **hata**

| | Doğrudan (`send_message`) | Yayın (`publish_message`) |
|---|---|---|
| Alıcı | tek adres | topic'e abone olan herkes |
| Dönüş değeri | **var** | **yok** |
| Handler çökerse | çağırana **fırlatır** | **loglanır, fırlatmaz** |

Son satır bir tasarım kararı: bir sonucu bekleyeceksen doğrudan, bir olayı
duyuracaksan yayın. Karıştırırsan hata sessizce kaybolur.

---

<a id="s4"></a>
## 4 · Tool döngüsü

<div align="center">
<svg viewBox="0 0 600 152" width="600" height="152"><path d="M17,53 L119,53 L119,95 L17,95 Z" fill="#ffffff" stroke="none"/><path d="M16.6,52.4 Q68.0,50.8 119.9,51.1 M121.0,52.0 Q119.5,74.0 120.0,95.2 M119.6,96.3 Q68.0,97.8 14.6,96.9 M15.1,94.5 Q15.5,74.0 17.6,52.8" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><path d="M16.4,50.5 Q68.0,53.4 119.5,52.0 M121.4,51.6 Q120.7,74.0 118.9,97.0 M121.4,95.9 Q68.0,97.4 16.2,95.5 M15.0,94.5 Q16.0,74.0 17.2,51.2" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><text x="68.0" y="77.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">AssistantAgent</text><path d="M177,53 L279,53 L279,95 L177,95 Z" fill="#e7f5ff" stroke="none"/><path d="M175.1,53.5 Q228.0,53.0 281.3,52.8 M280.3,52.7 Q279.5,74.0 278.8,95.9 M279.0,97.4 Q228.0,94.8 177.2,95.7 M175.8,95.6 Q175.8,74.0 177.6,51.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M174.7,52.7 Q228.0,50.3 281.6,52.0 M280.2,52.8 Q280.0,74.0 279.1,96.3 M280.4,95.0 Q228.0,95.8 174.9,95.5 M175.5,96.3 Q176.8,74.0 175.4,50.7" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="228.0" y="73.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">model</text><text x="228.0" y="84.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">create_stream</text><path d="M337,53 L439,53 L439,95 L337,95 Z" fill="#fff4e6" stroke="none"/><path d="M337.0,52.9 Q388.0,50.4 439.6,52.6 M441.3,53.5 Q439.7,74.0 439.7,97.6 M440.3,97.2 Q388.0,94.3 336.0,95.6 M337.2,94.9 Q335.6,74.0 334.6,53.0" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><path d="M335.9,53.1 Q388.0,53.0 438.6,51.2 M439.7,50.9 Q440.4,74.0 438.6,95.5 M441.5,96.8 Q388.0,94.6 335.9,95.2 M336.1,96.7 Q336.8,74.0 334.5,51.1" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><text x="388.0" y="73.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">workbench</text><text x="388.0" y="84.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">call_tool</text><path d="M487,53 L581,53 L581,95 L487,95 Z" fill="#ebfbee" stroke="none"/><path d="M486.9,53.2 Q534.0,53.2 582.3,51.8 M583.1,51.4 Q581.9,74.0 582.1,95.3 M580.4,96.6 Q534.0,95.0 487.5,94.5 M486.5,97.1 Q486.2,74.0 486.3,51.2" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M487.5,53.3 Q534.0,51.7 580.8,50.7 M580.5,51.8 Q582.7,74.0 580.9,97.2 M580.8,96.1 Q534.0,95.4 485.4,96.0 M484.9,95.0 Q485.5,74.0 485.6,52.7" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="534.0" y="77.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">sonuç</text><path d="M120.5,74.7 Q148.0,74.1 175.2,74.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M120.4,73.9 Q148.0,74.7 172.6,74.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M174.0,74.0 L167.3,78.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M174.0,74.0 L167.4,70.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M280.5,73.6 Q308.0,73.5 333.4,75.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M282.0,74.2 Q308.0,73.5 335.4,72.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M334.0,74.0 L327.9,78.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M334.0,74.0 L328.1,70.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="308.0" y="68.0" font-size="7.4" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" text-anchor="middle">tool isteği</text><path d="M442.6,74.1 Q463.0,73.6 484.2,72.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M443.5,73.5 Q463.0,74.6 483.4,75.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M484.0,74.0 L477.2,77.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M484.0,74.0 L477.2,70.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M535.8,52.4 Q518.5,33.5 498.5,16.5 M500.1,15.6 Q400.0,12.3 299.7,7.1 M299.8,7.2 Q210.2,17.2 120.7,23.0 M119.9,21.2 Q93.7,35.4 67.6,50.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M537.5,50.7 Q518.3,33.7 499.9,16.0 M501.2,16.9 Q400.0,11.0 300.4,8.8 M301.1,8.6 Q209.9,14.2 120.6,21.2 M119.6,22.4 Q93.4,34.8 69.0,51.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M68.0,50.0 L72.2,42.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M68.0,50.0 L75.2,50.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="280" y="6" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">döngü — max_tool_iterations</text><text x="16" y="122" font-size="8.4" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">VARSAYILAN 1: model tool sonucunu GÖRMEDEN cevap verir</text><text x="16" y="138" font-size="7.6" fill="#2f9e44" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">ölçüldü — bizde 6'ya çekildi</text></svg>
</div>

<sub>▲ Model tool ister · kapı · çalıştır · sonucu gör · döngü · düzenlemek için: [`f_tool_loop.excalidraw`](diagrams/wiki/f_tool_loop.excalidraw) → excalidraw.com'a sürükle</sub>


### Sessiz varsayılanlar — en pahalı tuzak

Ajan bir tool çağırdıktan sonra **kaç kez daha** dönebilir? Hiçbir çerçeve aynı
cevabı vermiyor, ve hiçbiri bunu öne çıkarmıyor. Hepsi kurulu paketten
okundu **[ölçüldü]**:

| Çerçeve | Alan | Varsayılan |
|---|---|---:|
| **AutoGen** | `max_tool_iterations` | **1** |
| OpenAI Agents SDK | `Runner.run(max_turns=)` | 10 |
| CrewAI | `Agent.max_iter` | 25 |
| **MAF** | `DEFAULT_MAX_ITERATIONS` | **40** |
| LangGraph | `recursion_limit` | 10007 |
| Google ADK | `LoopAgent.max_iterations` | **sınırsız** |

**AutoGen'de varsayılan 1:** ajan tool'u çağırır, sonucu görür ve **durur** —
cevabı hiç yazmaz. Hata da vermez.

> Tehlike iki uçta da aynı: **varsayılanı yazmadan koşturmak.** Bir uçta ajan
> sessizce hiçbir şey yapmıyor, öbür uçta sessizce durmuyor.

### Diğer sessiz varsayılanlar

* `model_context` verilmezse ajanın **belleği yok** — ve hata vermiyor.
* Sonlandırma koşulu yoksa takım **sonsuza kadar** konuşuyor; fatura gerçek.
* `description` boş bırakılan ajan, `SelectorGroupChat`'te **kör** seçiliyor.

---

<a id="s5"></a>
## 5 · Workbench

<div align="center">
<svg viewBox="0 0 600 176" width="600" height="176"><path d="M211,11 L389,11 L389,49 L211,49 Z" fill="#f8f0fc" stroke="none"/><path d="M210.3,9.4 Q300.0,10.0 391.4,10.7 M389.9,11.1 Q389.6,30.0 390.7,50.3 M389.8,51.5 Q300.0,51.4 210.2,49.6 M208.9,49.5 Q210.2,30.0 210.9,9.9" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M211.5,10.1 Q300.0,9.6 389.8,8.7 M390.0,9.3 Q389.8,30.0 391.4,50.4 M391.2,49.7 Q300.0,49.9 211.5,50.6 M209.9,51.5 Q209.8,30.0 210.3,8.4" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="300.0" y="29.2" font-size="8.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Workbench</text><text x="300.0" y="40.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">list_tools / call_tool</text><path d="M21,85 L179,85 L179,127 L21,127 Z" fill="#ebfbee" stroke="none"/><path d="M19.7,83.4 Q100.0,85.3 180.0,83.1 M180.4,83.9 Q179.7,106.0 179.9,127.5 M181.1,126.6 Q100.0,127.3 20.4,129.0 M20.9,129.4 Q20.9,106.0 20.4,85.1" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M19.2,84.7 Q100.0,84.9 180.4,84.1 M181.4,85.0 Q180.5,106.0 180.9,127.2 M179.0,127.6 Q100.0,129.2 19.5,128.8 M21.6,126.6 Q19.8,106.0 19.7,85.2" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="100.0" y="105.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">StaticWorkbench</text><text x="100.0" y="116.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">elindeki fonksiyonlar</text><path d="M221,85 L379,85 L379,127 L221,127 Z" fill="#e7f5ff" stroke="none"/><path d="M220.5,85.1 Q300.0,84.0 380.4,82.9 M381.5,83.7 Q380.8,106.0 380.1,126.8 M379.5,126.9 Q300.0,128.0 220.6,128.6 M221.1,129.5 Q220.2,106.0 221.1,82.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M220.5,85.2 Q300.0,84.1 380.1,83.8 M379.1,85.1 Q380.4,106.0 378.5,128.8 M378.6,127.0 Q300.0,126.3 219.3,129.0 M221.3,128.3 Q220.6,106.0 220.1,84.0" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="300.0" y="105.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">McpWorkbench</text><text x="300.0" y="116.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">stdio ya da HTTP</text><path d="M421,85 L579,85 L579,127 L421,127 Z" fill="#fff4e6" stroke="none"/><path d="M420.7,83.3 Q500.0,84.2 581.3,83.0 M579.9,82.5 Q579.2,106.0 578.7,128.5 M578.7,126.5 Q500.0,126.5 419.5,128.9 M419.7,127.3 Q419.5,106.0 418.4,82.7" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><path d="M420.4,84.7 Q500.0,83.5 580.5,85.3 M580.4,84.7 Q579.8,106.0 579.4,129.6 M581.3,127.9 Q500.0,129.3 419.1,127.5 M418.9,126.6 Q420.1,106.0 419.2,85.5" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><text x="500.0" y="105.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">GatedWorkbench</text><text x="500.0" y="116.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">bizim — kapı</text><path d="M99.5,82.6 Q100.4,68.0 99.1,52.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M99.5,83.5 Q99.9,68.0 98.7,54.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M100.0,54.0 L103.6,60.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M100.0,54.0 L96.2,60.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M298.8,83.4 Q300.2,68.0 301.4,54.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M300.6,81.8 Q300.3,68.0 299.3,53.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M300.0,54.0 L304.3,59.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M300.0,54.0 L296.0,59.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M501.1,82.4 Q499.4,68.0 500.2,54.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M500.2,81.4 Q500.6,68.0 501.5,52.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M500.0,54.0 L504.2,60.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M500.0,54.0 L495.4,59.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="20" y="148" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Tool tek bir arayüz; workbench BİR KOLEKSİYON — durum ve kaynak paylaşan tool'lar, tek tip sonuç.</text><text x="20" y="164" font-size="7.6" fill="#8a5208" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Ajan hangisiyle konuştuğunu bilmiyor. Kapıyı araya koymayı mümkün kılan tek şey bu.</text></svg>
</div>

<sub>▲ Üç kaynak, tek arayüz · düzenlemek için: [`f_workbench_component.excalidraw`](diagrams/wiki/f_workbench_component.excalidraw) → excalidraw.com'a sürükle</sub>


`tools=[...]` bir **liste**, `workbench=` bir **kaynak**. Liste ajan yazılırken
donuyor; kaynak her turda sorulabiliyor. İkisi birlikte kullanılamıyor —
`ValueError: Tools cannot be used with a workbench.`

**Her turda ne oluyor:**

```
wb.list_tools()  →  JSON şemalar  →  model çağrısına `tools=` diye gider
```

Model fonksiyonu görmüyor; **adını, tarifini ve parametre şemasını** görüyor.
Üç sonuç:

1. **Docstring gerçekten arayüz.** Modelin o tool'a *ne zaman* uzanacağına karar
   verdiği tek metin o.
2. **Şemalar her turda ödeniyor.** 17 tool = her istekte 17 şema.
3. **Bir tool'u listeden çıkarmak** prompt'u ucuzlatıyor — *kapılamak* ile
   *filtrelemek* ayrı kararlar.

**Neden kapıyı buraya koyduk:** workbench, yerel bir Python fonksiyonuyla uzak
bir MCP tool'unu **aynı gören tek yer**. Ve kural, ajan yazılırken **var olmayan**
tool'lar için de geçerli — "şu isimler tehlikeli" listesi tam burada başarısız
olurdu.

---

<a id="s6"></a>
## 6 · Onay kapısı

<div align="center">
<svg viewBox="0 0 600 158" width="600" height="158"><path d="M17,47 L111,47 L111,87 L17,87 Z" fill="#e7f5ff" stroke="none"/><path d="M15.1,45.4 Q64.0,44.5 113.3,46.0 M110.7,46.0 Q111.6,67.0 113.1,88.1 M113.2,87.6 Q64.0,86.4 14.4,88.8 M14.4,88.7 Q15.4,67.0 17.1,47.1" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M15.0,44.7 Q64.0,47.6 110.8,47.5 M110.9,44.5 Q112.7,67.0 110.6,86.5 M110.9,87.4 Q64.0,88.5 17.4,86.9 M16.8,88.2 Q16.5,67.0 16.2,46.9" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="64.0" y="70.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">ajan</text><path d="M159,35 L289,35 L289,97 L159,97 Z" fill="#fff4e6" stroke="none"/><path d="M157.1,33.5 Q224.0,34.3 289.2,33.7 M289.8,32.6 Q288.8,66.0 289.9,99.0 M290.6,96.7 Q224.0,98.2 156.6,98.5 M157.4,97.6 Q156.8,66.0 157.0,32.5" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><path d="M159.4,34.6 Q224.0,32.2 291.4,34.0 M291.2,34.4 Q288.8,66.0 290.1,98.0 M289.6,98.7 Q224.0,99.0 157.0,99.1 M158.3,99.0 Q158.0,66.0 158.5,32.8" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><text x="224.0" y="65.2" font-size="8.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">GatedWorkbench</text><text x="224.0" y="76.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">call_tool</text><path d="M112.4,64.5 Q135.0,65.5 157.4,66.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M114.9,65.4 Q135.0,67.0 157.5,67.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M156.0,66.0 L149.7,69.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M156.0,66.0 L149.5,61.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M345,13 L455,13 L455,49 L345,49 Z" fill="#ebfbee" stroke="none"/><path d="M342.7,12.7 Q400.0,11.5 457.5,11.1 M456.3,11.9 Q455.6,31.0 457.3,49.6 M454.5,51.5 Q400.0,48.8 343.0,48.4 M344.7,50.6 Q344.6,31.0 344.9,11.6" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M343.4,13.5 Q400.0,10.6 457.5,11.1 M455.8,13.3 Q456.1,31.0 454.4,51.1 M456.2,48.8 Q400.0,50.9 342.7,51.2 M345.5,51.2 Q344.0,31.0 344.8,10.6" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="400.0" y="30.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">İZİN</text><text x="400.0" y="41.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">tool koşar</text><path d="M345,83 L455,83 L455,119 L345,119 Z" fill="#fff5f5" stroke="none"/><path d="M345.6,82.0 Q400.0,82.3 454.6,81.0 M457.5,81.2 Q455.3,101.0 455.4,120.7 M455.0,121.6 Q400.0,120.0 344.4,119.6 M344.0,120.8 Q343.3,101.0 343.0,80.7" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><path d="M345.3,81.6 Q400.0,82.8 455.9,82.4 M454.8,81.3 Q456.0,101.0 455.1,119.0 M455.7,119.1 Q400.0,121.5 345.1,119.4 M345.0,119.3 Q344.7,101.0 342.9,82.2" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><text x="400.0" y="100.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">RET</text><text x="400.0" y="111.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">gerekçe döner</text><path d="M292.1,54.8 Q317.0,43.9 342.1,32.8" fill="none" stroke="#2f9e44" stroke-width="1.4" stroke-linecap="round"/><path d="M291.2,57.4 Q316.9,43.9 341.3,33.6" fill="none" stroke="#2f9e44" stroke-width="1.4" stroke-linecap="round"/><path d="M342.0,32.0 L338.5,37.8" fill="none" stroke="#2f9e44" stroke-width="1.4" stroke-linecap="round"/><path d="M342.0,32.0 L334.5,30.4" fill="none" stroke="#2f9e44" stroke-width="1.4" stroke-linecap="round"/><path d="M291.7,76.7 Q316.7,88.7 343.4,99.0" fill="none" stroke="#c92a2a" stroke-width="1.4" stroke-linecap="round"/><path d="M292.2,76.6 Q316.9,88.3 343.5,97.4" fill="none" stroke="#c92a2a" stroke-width="1.4" stroke-linecap="round"/><path d="M342.0,98.0 L334.7,99.6" fill="none" stroke="#c92a2a" stroke-width="1.4" stroke-linecap="round"/><path d="M342.0,98.0 L337.7,91.1" fill="none" stroke="#c92a2a" stroke-width="1.4" stroke-linecap="round"/><path d="M491,83 L585,83 L585,119 L491,119 Z" fill="#f8f0fc" stroke="none"/><path d="M490.0,81.7 Q538.0,83.4 587.2,83.2 M587.3,81.5 Q585.7,101.0 584.7,119.5 M586.9,120.8 Q538.0,119.6 490.4,120.3 M490.2,118.4 Q490.3,101.0 490.1,81.3" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M490.4,83.1 Q538.0,80.5 585.3,82.5 M586.6,83.2 Q585.8,101.0 586.6,118.7 M585.5,119.7 Q538.0,119.6 490.6,119.0 M491.4,118.6 Q490.2,101.0 491.1,82.7" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="538.0" y="100.2" font-size="7.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">onay isteği</text><text x="538.0" y="111.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">id + argüman</text><path d="M458.9,98.7 Q473.0,99.9 486.9,101.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M456.9,98.8 Q473.0,100.2 487.8,100.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M488.0,100.0 L482.5,103.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M488.0,100.0 L482.0,95.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="16" y="130" font-size="8" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Onay metni id'yi TAŞIMALI — düşerse arayüz düğmeyi çizemez.</text><text x="16" y="146" font-size="7.6" fill="#2f9e44" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Bu hatayı ölçtük: testi önce eski koda karşı düşürdük.</text></svg>
</div>

<sub>▲ Çağrı geçmeden önce duran tek nokta · düzenlemek için: [`f_gate.excalidraw`](diagrams/wiki/f_gate.excalidraw) → excalidraw.com'a sürükle</sub>


### Üç kural

**① Engellenen çağrı hata *döndürüyor*, fırlatmıyor.** Ajan reddedildiğini
öğreniyor, söyleyebiliyor, başka yol deneyebiliyor. İstisna turu bitirir ve
insana hiçbir şey anlatmazdı.

**② Onay bir kez tüketiliyor.** İmza `(tool, argümanlar)` üstünde. Aynı çağrı
ikinci kez geldiğinde **yeniden soruluyor**. "Bir daha sorma" bir kolaylık
kararıdır ve düzenlenmiş bir kurumda varsayılanı açık olmamalıdır.

**③ Bozulan bekçi kapanır, açılmaz.** Kanca kendi istisnasında `block: True`
döndürüyor.

### Kapılamak ≠ filtrelemek

| | Ne yapar | Ne zaman doğru |
|---|---|---|
| **Kapılamak** | tool görünür kalır, çağrı reddedilir | ajan *"mesaj atardım ama onayınız lazım"* diyebilir |
| **Filtrelemek** | `list_tools`'tan çıkar, prompt'a hiç girmez | prompt maliyeti · meşru kullanımı olmayan tool |

Filtrelenmiş tool **adıyla çağrılsa da reddediliyor** — *liste bir ipucudur,
zorlama noktası değil.*

---

<a id="s7"></a>
## 7 · Takımlar

<div align="center">
<svg viewBox="0 0 600 128" width="600" height="128"><path d="M13,27 L115,27 L115,77 L13,77 Z" fill="#f8f0fc" stroke="none"/><path d="M10.8,24.5 Q64.0,27.6 117.6,25.0 M116.5,25.5 Q116.8,52.0 117.2,77.1 M115.4,78.3 Q64.0,76.3 13.4,78.6 M12.7,76.6 Q12.9,52.0 13.2,26.3" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M11.0,27.1 Q64.0,25.3 116.3,27.4 M117.5,26.2 Q115.4,52.0 116.6,78.4 M117.5,77.4 Q64.0,78.9 12.9,79.5 M10.9,79.0 Q12.3,52.0 12.5,25.0" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="64.0" y="51.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">RoundRobin</text><text x="64.0" y="62.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">sırayla</text><path d="M131,27 L233,27 L233,77 L131,77 Z" fill="#f8f0fc" stroke="none"/><path d="M130.1,24.5 Q182.0,24.4 235.4,26.0 M233.7,26.5 Q234.0,52.0 235.3,77.4 M234.3,77.4 Q182.0,78.5 130.3,78.1 M130.6,78.1 Q129.9,52.0 128.7,27.6" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M129.8,25.7 Q182.0,25.5 235.4,27.4 M233.7,24.7 Q234.0,52.0 235.3,76.9 M233.5,77.5 Q182.0,77.8 129.5,77.0 M130.1,78.9 Q129.0,52.0 130.5,25.0" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="182.0" y="51.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Selector</text><text x="182.0" y="62.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">model seçer</text><path d="M249,27 L351,27 L351,77 L249,77 Z" fill="#f8f0fc" stroke="none"/><path d="M246.6,25.7 Q300.0,26.7 353.0,27.3 M353.1,26.0 Q352.1,52.0 352.1,78.3 M352.8,77.9 Q300.0,76.2 248.7,78.5 M248.9,79.0 Q247.7,52.0 248.3,27.4" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M248.0,26.4 Q300.0,24.3 351.5,24.9 M351.8,25.3 Q351.6,52.0 350.5,79.5 M350.8,77.1 Q300.0,78.5 247.2,79.2 M247.3,76.7 Q247.3,52.0 247.9,25.1" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="300.0" y="51.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">Swarm</text><text x="300.0" y="62.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">handoff</text><path d="M367,27 L469,27 L469,77 L367,77 Z" fill="#f8f0fc" stroke="none"/><path d="M366.8,27.4 Q418.0,25.4 471.1,24.5 M471.1,24.5 Q470.3,52.0 469.1,78.0 M470.8,78.7 Q418.0,78.4 365.9,77.2 M365.3,76.6 Q365.4,52.0 366.5,24.9" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M366.5,24.6 Q418.0,27.4 470.0,26.3 M470.0,24.5 Q471.0,52.0 470.9,76.4 M471.6,79.1 Q418.0,77.5 365.1,78.1 M365.5,79.2 Q365.3,52.0 367.0,24.4" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="418.0" y="51.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">MagenticOne</text><text x="418.0" y="62.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">planlayıcı</text><path d="M485,27 L587,27 L587,77 L485,77 Z" fill="#f8f0fc" stroke="none"/><path d="M484.4,25.0 Q536.0,25.6 589.0,27.3 M587.6,27.1 Q587.9,52.0 588.0,77.0 M586.9,79.1 Q536.0,78.8 484.0,78.6 M485.5,77.0 Q484.7,52.0 483.7,27.2" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M483.6,26.5 Q536.0,26.9 587.0,24.9 M587.6,25.2 Q588.9,52.0 588.1,79.5 M588.2,79.2 Q536.0,79.2 484.3,78.1 M482.5,79.4 Q484.6,52.0 484.6,27.0" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="536.0" y="51.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">GraphFlow</text><text x="536.0" y="62.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">DAG</text><text x="12" y="100" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Beşi de aynı arayüz: run() / run_stream() → TaskResult</text><text x="12" y="116" font-size="7.6" fill="#2f9e44" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Taramamız GraphFlow kullanıyor — eşzamanlı dal + join(all)</text></svg>
</div>

<sub>▲ Beş takım tipi — değişen tek şey: sırayı kim belirliyor · düzenlemek için: [`f_teams.excalidraw`](diagrams/wiki/f_teams.excalidraw) → excalidraw.com'a sürükle</sub>


Aynı görev, aynı ajanlar, yalnız orkestrasyon değişiyor **[ölçüldü]**:

| Desen | Sırayı kim belirliyor | Mesaj | LLM | Tool | Token |
|---|---|---:|---:|---:|---:|
| **SelectorGroupChat** | model her turda seçiyor | 8 | 5 | 2 | **204** |
| GraphFlow | önceden çizilmiş DAG | 11 | 7 | 3 | 270 |
| RoundRobinGroupChat | sırayla, kararsız | 9 | 6 | 2 | 274 |
| **Swarm** (handoff) | ajanın kendisi devrediyor | 14 | 7 | 4 | **334** |

**%63,7 fark.** Ödenen şey zekâ değil **yönlendirme özerkliği**: ajanlara
"kime devredeceğine sen karar ver" dediğin an fatura artıyor, çünkü her devir
bir tur ve her tur bir model çağrısı.

> Kıyasa çevirisi: **Agents SDK'nın tek modeli olan handoff, AutoGen'in en
> pahalı desenidir.** Tek desenli bir çerçeve seçmek, o desenin faturasını da
> seçmektir.

---

<a id="s8"></a>
## 8 · Kod yürütme

<div align="center">
<svg viewBox="0 0 600 150" width="600" height="150"><path d="M17,31 L133,31 L133,73 L17,73 Z" fill="#f8f9fa" stroke="none"/><path d="M14.7,29.7 Q75.0,28.4 132.7,31.3 M134.1,29.5 Q134.8,52.0 135.1,72.9 M133.5,73.2 Q75.0,74.6 14.4,73.8 M16.3,72.6 Q15.3,52.0 15.4,29.8" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><path d="M14.7,28.9 Q75.0,31.5 134.9,28.5 M134.2,29.2 Q133.3,52.0 135.1,74.9 M135.0,73.1 Q75.0,74.6 16.3,74.1 M15.4,73.7 Q16.5,52.0 16.9,30.0" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><text x="75.0" y="51.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">kod bloğu</text><text x="75.0" y="62.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">modelden</text><path d="M136.4,52.5 Q154.0,52.3 173.2,52.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M135.6,50.5 Q154.0,52.3 172.8,51.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M172.0,52.0 L166.2,56.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M172.0,52.0 L166.0,47.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M175,9 L363,9 L363,47 L175,47 Z" fill="#fff5f5" stroke="none"/><path d="M173.7,6.6 Q269.0,7.4 363.9,8.2 M363.0,6.7 Q364.3,28.0 364.4,46.4 M363.1,48.1 Q269.0,48.5 174.7,49.3 M175.6,49.5 Q174.2,28.0 174.4,7.1" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><path d="M173.0,9.6 Q269.0,7.9 363.8,8.8 M362.8,7.0 Q364.7,28.0 363.2,47.5 M363.8,47.7 Q269.0,48.8 172.7,46.6 M172.7,47.8 Q173.8,28.0 172.4,8.1" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><text x="269.0" y="27.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">LocalCommandLine…</text><text x="269.0" y="38.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">host makinede</text><path d="M175,61 L363,61 L363,99 L175,99 Z" fill="#ebfbee" stroke="none"/><path d="M174.6,59.7 Q269.0,58.5 365.6,59.2 M364.8,59.5 Q364.0,80.0 363.5,100.4 M365.2,99.7 Q269.0,98.3 174.7,100.6 M173.2,98.7 Q173.8,80.0 175.2,58.6" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M174.1,58.9 Q269.0,60.1 364.5,61.3 M363.6,58.8 Q363.6,80.0 365.5,100.1 M365.0,99.6 Q269.0,99.5 172.6,98.8 M173.5,101.2 Q174.3,80.0 174.8,61.5" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="269.0" y="79.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">DockerCommandLine…</text><text x="269.0" y="90.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">konteynerde</text><path d="M365.4,28.9 Q384.0,28.3 400.5,28.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M366.0,26.8 Q384.0,28.7 402.0,28.1" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M402.0,28.0 L395.6,32.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M402.0,28.0 L395.6,23.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M365.1,79.4 Q384.0,80.0 401.7,81.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M365.0,79.0 Q384.0,79.3 403.6,81.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M402.0,80.0 L395.3,84.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M402.0,80.0 L395.2,75.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M405,9 L587,9 L587,47 L405,47 Z" fill="#fff5f5" stroke="none"/><path d="M404.1,9.4 Q496.0,7.4 589.0,8.1 M587.9,8.7 Q587.5,28.0 588.9,47.2 M588.2,48.0 Q496.0,46.8 403.7,47.1 M402.8,48.4 Q404.0,28.0 405.3,8.9" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><path d="M404.0,8.6 Q496.0,8.4 586.8,6.6 M587.6,8.5 Q587.4,28.0 587.7,47.5 M587.5,48.8 Q496.0,47.0 405.0,47.0 M402.6,47.7 Q404.8,28.0 404.0,9.1" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><text x="496.0" y="31.2" font-size="7.8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">→ host'un her şeyi</text><path d="M405,61 L587,61 L587,99 L405,99 Z" fill="#ebfbee" stroke="none"/><path d="M404.2,59.6 Q496.0,58.8 589.3,58.8 M589.0,59.4 Q588.4,80.0 589.4,98.6 M586.7,100.2 Q496.0,100.4 405.5,99.4 M403.3,99.8 Q404.2,80.0 404.2,59.2" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M404.0,61.3 Q496.0,60.7 587.3,61.5 M586.5,60.5 Q588.5,80.0 588.7,99.9 M589.6,98.9 Q496.0,101.6 405.2,101.3 M405.2,99.9 Q403.6,80.0 405.3,61.2" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="496.0" y="83.2" font-size="7.8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">→ yalıtılmış</text><text x="16" y="122" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Her kod bloğu bir dosyaya yazılıp AYRI BİR SÜREÇTE koşuyor — yani bloklar arası değişken paylaşımı yok.</text><text x="16" y="138" font-size="7.6" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Local, modelin yazdığı kodu makinende koşturur. Bu bir tercih değil, bir güven kararıdır.</text></svg>
</div>

<sub>▲ Yerel yürütücü ve konteyner · düzenlemek için: [`f_code_executors.excalidraw`](diagrams/wiki/f_code_executors.excalidraw) → excalidraw.com'a sürükle</sub>


### Rol: yirmi ikinci tool değil, **kaçış kapağı**

Model önce mevcut tool'lara bakıyor; sorulanı karşılayan bir tool **yoksa**
Python yazıp çalıştırıyor. Ayrım tarifle zorlanıyor: tarif *"kod çalıştırır"*
deseydi ajan her hesabı yeniden icat eder, yirmi bir tool boşa çalışırdı.

### Ömür: konteyner **sürece** ait, çağrıya değil

Sunucu açılırken bir konteyner kalkıyor, kapanırken iniyor. Çağrı başına
konteyner kaldırmak 2–3 saniye ve bunun tamamı kullanıcının beklediği süreye
eklenirdi.

**Bedeli:** konteyner turlar arasında **durum taşıyor**. İzolasyon konteyner ile
host arasında; tur ile tur arasında değil.

### Güvenlik — ölçüldü, ve iyi görünmüyor

| | Değer |
|---|---|
| kullanıcı | **root** (uid=0) |
| ağ | **bridge** — dışarı çıkıyor (pypi.org'a `200` alındı) |
| salt okunur kök | hayır |
| bellek / CPU / PID sınırı | **yok** |
| düşürülen yetki | **hiçbiri** |
| ayrıcalıklı | hayır ✔ |

Hiçbiri tercih değil: `DockerCommandLineCodeExecutor`'da bu parametrelerin
**hiçbiri yok**.

**Buna karşılık:** varsayılan kapalı · her koşuda insan onayı · onay kartı ağ
erişimini açıkça yazıyor · onay **kodun imzasına** bağlı · 60 sn zaman aşımı.

> Gerçek savunma sandbox değil, **kapı**. Bu wiki'de *"sandbox güvenli"* cümlesi
> kurulmuyor.

### Onay neden saklanan metni koşturuyor

Kapının reddi turu bitiriyor. Onay o turu geri getiremiyor, ve modelden kodu
yeniden istemek işe yaramıyor — **ölçüldü: aynı soru iki farklı program üretti**
(imzalar `029f4d1f…` ve `107fdfd1…`). Onaylananla çalışanın aynı olmasının tek
yolu, çalıştırılacak olanın **onaylanan metin** olması.

---

<a id="s9"></a>
## 9 · Zamanlayıcı

<div align="center">
<svg viewBox="0 0 600 152" width="600" height="152"><path d="M9,9 L135,9 L135,123 L9,123 Z" fill="#f8f9fa" stroke="none"/><path d="M8.4,6.4 Q72.0,7.4 137.5,9.0 M135.2,7.0 Q134.2,66.0 134.6,123.0 M136.6,123.6 Q72.0,125.7 9.0,124.8 M6.9,123.5 Q7.5,66.0 6.5,8.5" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><path d="M6.9,9.2 Q72.0,6.9 135.8,9.4 M136.5,7.2 Q134.3,66.0 134.9,123.2 M136.0,123.4 Q72.0,123.1 6.5,123.5 M6.6,123.2 Q7.6,66.0 7.3,7.7" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><text x="72.0" y="69.2" font-size="9.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle"></text><text x="72" y="26" font-size="8.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">TETİKLEYİCİ</text><text x="18" y="44" font-size="6.6" fill="#454c53" font-family="DejaVu Sans Mono, monospace">at · tek sefer</text><text x="18" y="57" font-size="6.6" fill="#454c53" font-family="DejaVu Sans Mono, monospace">every · aralık</text><text x="18" y="70" font-size="6.6" fill="#454c53" font-family="DejaVu Sans Mono, monospace">cron · ifade</text><text x="18" y="83" font-size="6.6" fill="#454c53" font-family="DejaVu Sans Mono, monospace">on-exit · komut</text><text x="18" y="96" font-size="6.6" fill="#454c53" font-family="DejaVu Sans Mono, monospace">stream · satır</text><text x="18" y="109" font-size="6.6" fill="#454c53" font-family="DejaVu Sans Mono, monospace">webhook · dışarıdan</text><path d="M161,9 L291,9 L291,59 L161,59 Z" fill="#e7f5ff" stroke="none"/><path d="M160.2,9.1 Q226.0,8.0 291.3,6.7 M291.1,8.2 Q292.4,34.0 291.9,59.4 M292.8,60.4 Q226.0,61.5 160.4,60.6 M158.5,59.0 Q160.7,34.0 159.9,8.6" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M160.0,7.4 Q226.0,8.0 291.2,9.2 M292.0,7.9 Q291.0,34.0 290.8,60.3 M292.4,59.3 Q226.0,61.0 159.9,61.2 M160.1,59.4 Q159.6,34.0 160.0,9.2" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="226.0" y="33.2" font-size="8.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">AUTOMATIONS</text><text x="226.0" y="44.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">tam zamanlama</text><path d="M161,73 L291,73 L291,123 L161,123 Z" fill="#f8f0fc" stroke="none"/><path d="M161.5,71.7 Q226.0,72.0 291.9,72.5 M292.6,71.3 Q292.0,98.0 290.4,123.8 M291.2,124.3 Q226.0,125.6 160.6,125.3 M160.2,123.5 Q160.4,98.0 159.9,71.3" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M160.1,72.4 Q226.0,70.6 291.4,72.8 M291.4,73.1 Q291.9,98.0 290.5,123.3 M292.6,123.2 Q226.0,124.6 161.5,122.7 M159.9,124.8 Q159.9,98.0 159.2,70.9" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="226.0" y="97.2" font-size="8.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">HEARTBEAT</text><text x="226.0" y="108.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">~30 dk · main oturum</text><path d="M138.9,40.2 Q148.1,37.2 159.3,33.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M138.0,38.8 Q148.0,37.0 159.0,32.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M158.0,34.0 L152.6,40.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M158.0,34.0 L150.8,31.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M137.0,83.9 Q147.8,90.4 156.4,94.9" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M138.6,82.8 Q148.0,90.0 158.8,96.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M158.0,96.0 L150.8,97.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M158.0,96.0 L154.8,88.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M317,9 L439,9 L439,59 L317,59 Z" fill="#ebfbee" stroke="none"/><path d="M317.2,6.7 Q378.0,6.4 439.3,8.5 M440.8,8.2 Q439.5,34.0 439.4,61.3 M439.2,61.6 Q378.0,61.7 315.0,61.1 M316.4,60.4 Q316.2,34.0 315.8,9.1" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M315.3,8.9 Q378.0,6.4 440.8,8.5 M439.3,8.3 Q439.6,34.0 440.2,61.0 M439.3,59.4 Q378.0,61.0 315.1,60.7 M314.6,61.3 Q316.0,34.0 316.1,9.0" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="378.0" y="33.2" font-size="8.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">TASK KAYDI</text><text x="378.0" y="44.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">queued→running→…</text><path d="M293.7,33.4 Q304.0,34.2 314.2,35.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M295.5,34.2 Q304.0,34.3 314.1,33.4" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M314.0,34.0 L307.8,38.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M314.0,34.0 L308.1,30.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="304.0" y="28.0" font-size="7.4" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" text-anchor="middle">her koşu</text><path d="M293.2,98.5 Q337.0,99.7 378.6,97.3" fill="none" stroke="#c92a2a" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="4 3"/><path d="M292.4,99.2 Q337.0,98.4 380.1,97.2" fill="none" stroke="#c92a2a" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="4 3"/><text x="300" y="112" font-size="6.8" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">heartbeat task ÜRETMEZ</text><path d="M465,9 L587,9 L587,59 L465,59 Z" fill="#fff4e6" stroke="none"/><path d="M464.8,6.7 Q526.0,7.7 589.3,8.5 M587.0,8.4 Q587.9,34.0 587.1,58.6 M587.5,60.9 Q526.0,58.7 463.9,60.2 M465.0,60.9 Q463.8,34.0 463.6,8.2" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><path d="M465.3,7.9 Q526.0,7.2 588.3,8.2 M587.5,8.9 Q588.8,34.0 588.3,59.2 M589.2,61.0 Q526.0,61.2 465.2,59.0 M463.8,59.8 Q464.1,34.0 463.6,8.5" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><text x="526.0" y="33.2" font-size="8.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">KUYRUK</text><text x="526.0" y="44.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">lane · FIFO</text><path d="M440.6,32.5 Q452.0,34.1 461.0,35.0" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M442.5,33.6 Q452.0,34.4 462.1,34.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M462.0,34.0 L455.3,38.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M462.0,34.0 L456.4,29.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M465,77 L587,77 L587,117 L465,117 Z" fill="#ffffff" stroke="none"/><path d="M463.3,77.2 Q526.0,75.1 589.3,77.4 M588.1,75.0 Q588.1,97.0 586.5,118.4 M589.2,119.4 Q526.0,117.8 464.1,116.6 M465.4,117.9 Q464.1,97.0 464.5,76.0" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><path d="M463.7,75.1 Q526.0,75.8 587.8,76.9 M586.5,75.1 Q587.6,97.0 586.6,116.7 M588.7,117.0 Q526.0,119.8 464.5,119.1 M463.2,116.8 Q463.4,97.0 462.8,77.5" fill="none" stroke="#1e1e1e" stroke-width="1.6" stroke-linecap="round"/><text x="526.0" y="96.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">yürütme</text><text x="526.0" y="107.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">izole / paylaşılan</text><path d="M527.5,61.0 Q525.8,68.0 525.8,74.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M526.5,63.2 Q526.3,68.0 527.4,75.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M526.0,74.0 L521.5,68.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><path d="M526.0,74.0 L529.8,67.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round"/><text x="8" y="140" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Sıra soldan sağa: ne tetikler → kim karar verir → ne kaydedilir → nasıl serileşir → nerede koşar.</text></svg>
</div>

<sub>▲ Zamanlama yığını · düzenlemek için: [`f_task_stack.excalidraw`](diagrams/wiki/f_task_stack.excalidraw) → excalidraw.com'a sürükle</sub>


**AutoGen'de zamanlama diye bir kavram yok** — ve bu bir eksiklik değil, bir
kütüphane saat tutmaz.

Bizde iki katman var, biri bağlı:

* **Çevirmen (bağlı).** Türkçe "ne zaman" ifadesini cron şekline çeviriyor.
  Üç biçim kabul ediyor — `her gün 09:00` · `30dk` · `20dk sonra` — ve
  dördüncüsünü **tahmin etmiyor**, sözdizimini yazıp reddediyor.
* **Yerli zamanlayıcı (yazıldı, bağlanmadı).** 322 satır, 19 test.

### Üç bilinçli kısıt

| Karar | Neden |
|---|---|
| Payload hep `agentTurn` | `command`/`script` de var ama **ikisi de kabuk**; kabuk kararı onay kapısına ait, gece 3'te koşan bir iş tanımına değil |
| `sessionTarget: isolated` | Zamanlanmış koşu birinin konuşmasını ne miras almalı ne kirletmeli |
| `to` asla varsayılan değil | Adres tahmin etmek, yabancıya mektup atmak |

**Kapı yazılanı imzalıyor, çözülmüş zamanı değil.** `"20dk sonra"` her
ayrıştırmada başka bir damga veriyor; sonucun üstündeki imza hiç tutmazdı.

### Dürüst sınır

Zamanlama yalnız OpenClaw'ın Gateway'i koşarken çalışıyor. Sessizce ateşlemeyi
bırakmış bir iş, bir zamanlayıcının en kötü arızası — o yüzden liste, Gateway'e
ulaşılamamasını **boş liste değil, kendi durumu** olarak raporluyor.

---

<a id="s10"></a>
## 10 · OpenClaw'dan alınanlar

<div align="center">
<svg viewBox="0 0 600 174" width="600" height="174"><path d="M15,31 L191,31 L191,91 L15,91 Z" fill="#e7f5ff" stroke="none"/><path d="M14.0,28.9 Q103.0,29.1 193.0,29.1 M193.4,29.7 Q191.9,61.0 193.5,92.7 M192.4,92.3 Q103.0,91.8 13.6,93.6 M15.5,93.1 Q13.8,61.0 12.5,29.2" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M15.0,29.0 Q103.0,29.9 192.4,29.6 M193.4,30.8 Q191.3,61.0 191.0,92.8 M193.1,91.9 Q103.0,92.2 15.3,93.6 M15.0,93.0 Q14.8,61.0 14.3,29.0" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="103.0" y="60.2" font-size="9" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">SANDBOX</text><text x="103.0" y="71.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">tool NEREDE koşar</text><path d="M211,31 L387,31 L387,91 L211,91 Z" fill="#fff4e6" stroke="none"/><path d="M209.9,28.5 Q299.0,31.8 387.8,29.7 M389.5,28.8 Q388.7,61.0 389.3,92.2 M387.5,91.8 Q299.0,93.2 208.8,92.8 M211.5,93.1 Q210.4,61.0 210.9,31.6" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><path d="M210.5,30.1 Q299.0,31.6 388.1,31.0 M389.1,31.1 Q386.9,61.0 386.5,91.7 M388.3,92.3 Q299.0,93.5 210.2,93.1 M210.7,92.3 Q209.3,61.0 209.1,31.5" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><text x="299.0" y="60.2" font-size="9" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">TOOL POLICY</text><text x="299.0" y="71.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">HANGİ tool çağrılır</text><path d="M407,31 L583,31 L583,91 L407,91 Z" fill="#fff5f5" stroke="none"/><path d="M405.9,30.1 Q495.0,28.4 584.3,31.5 M585.2,29.3 Q583.9,61.0 583.9,90.9 M583.8,92.9 Q495.0,91.5 404.8,91.9 M404.4,92.5 Q407.1,61.0 406.6,30.7" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><path d="M406.0,31.5 Q495.0,30.2 583.3,30.7 M584.7,29.9 Q583.6,61.0 582.5,92.9 M584.9,93.6 Q495.0,92.8 406.9,91.7 M405.6,91.4 Q406.9,61.0 406.2,30.5" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><text x="495.0" y="60.2" font-size="9" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">ELEVATED</text><text x="495.0" y="71.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">sandbox'tan KAÇIŞ</text><text x="24" y="108" font-size="7.4" fill="#767d84" font-family="DejaVu Sans Mono, monospace">sandbox.mode</text><text x="220" y="108" font-size="7.4" fill="#767d84" font-family="DejaVu Sans Mono, monospace">tools.allow / deny</text><text x="416" y="108" font-size="7.4" fill="#767d84" font-family="DejaVu Sans Mono, monospace">tools.elevated.*</text><text x="14" y="134" font-size="8.4" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">deny HER ZAMAN kazanır · allow doluysa gerisi kapalı</text><text x="14" y="150" font-size="8.4" fill="#8a5208" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">Ama: tool policy ADA göre filtreler — exec'in İÇİNİ görmez.</text><text x="14" y="164" font-size="8.4" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">"write'ı kapattık, artık read-only" YANLIŞTIR.</text></svg>
</div>

<sub>▲ Üç kontrol ekseni — karıştırmak en yaygın hata · düzenlemek için: [`f_three_axes.excalidraw`](diagrams/wiki/f_three_axes.excalidraw) → excalidraw.com'a sürükle</sub>


"İzin" tek kavram değil, **üç ayrı soru**:

| Eksen | Soru |
|---|---|
| **Sandbox** | Tool **nerede** koşuyor? |
| **Tool policy** | **Hangi** tool çağrılabilir? |
| **Elevated** | Kutunun **dışına çıkış** var mı? |

Kurallar: `deny` her zaman kazanır · `allow` doluysa listede olmayan her şey
bloklu · tool policy sert duraktır.

**Ve OpenClaw'ın kendi belgesindeki uyarı:**

> *"Tool policy tool'u **adına göre** filtreler; `exec` içindeki yan etkileri
> incelemez. `exec` serbestse, `write`/`edit`'i reddetmek shell komutlarını
> salt-okunur yapmaz."*

Yani **"yazma tool'unu kapattık, artık read-only" cümlesi yanlıştır.**

### Taşınacak fikir: rol bir tool listesi değil, **grup adı**

OpenClaw'da 13 tool grubu var (`group:fs`, `group:runtime`, `group:web`…).
KKB'de bu `group:musteri-verisi`, `group:kredi-sorgu`, `group:rapor`,
`group:dis-erisim` olur. Yeni bir tool eklendiğinde **40 rol dosyası
güncellenmiyor**.

### Diğer alınanlar

* **Onay komuta değil, plana bağlanır** — donmuş plan.
* **Dış içerik veri, talimat değil.**
* **Kademeli açığa çıkarma:** prompt'a yalnız bir satırlık tarif giriyor,
  gövde ancak seçilince ödeniyor.

---

<a id="s11"></a>
## 11 · Denetim

<div align="center">
<svg viewBox="0 0 600 182" width="600" height="182"><path d="M15,25 L287,25 L287,115 L15,115 Z" fill="#e7f5ff" stroke="none"/><path d="M12.8,24.2 Q151.0,25.5 287.9,25.5 M287.8,24.2 Q288.4,70.0 287.6,117.3 M288.7,114.6 Q151.0,114.6 15.6,116.2 M14.1,116.1 Q14.4,70.0 15.4,23.3" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M15.6,23.2 Q151.0,22.7 287.3,23.8 M287.6,22.4 Q286.6,70.0 287.3,115.1 M289.3,116.7 Q151.0,117.7 12.8,115.2 M14.6,115.8 Q15.7,70.0 14.3,25.5" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="151.0" y="39" font-size="9.2" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">OPERASYONEL</text><text x="28" y="60" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">best-effort · yalnız metadata</text><text x="28" y="76" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">30 gün · 100.000 satır</text><text x="28" y="92" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">kuyruk dolarsa DÜŞER, koşu sürer</text><path d="M313,25 L585,25 L585,115 L313,115 Z" fill="#ebfbee" stroke="none"/><path d="M312.4,22.5 Q449.0,23.1 586.0,23.2 M585.7,23.8 Q586.6,70.0 587.3,115.6 M587.1,117.4 Q449.0,117.7 313.1,115.3 M312.9,114.7 Q311.5,70.0 310.6,24.0" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M311.9,22.9 Q449.0,23.0 587.5,24.0 M585.4,23.3 Q586.6,70.0 586.9,117.0 M587.4,116.3 Q449.0,115.0 311.6,115.7 M311.6,116.0 Q310.7,70.0 313.2,24.2" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="449.0" y="39" font-size="9.2" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">UYUM ARŞİVİ</text><text x="326" y="60" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">kayıpsız · senkron</text><text x="326" y="76" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">denetçiye gösterilir</text><text x="326" y="92" font-size="7.6" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">yazılamazsa KOŞU DÜŞER</text><text x="14" y="140" font-size="8.4" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">OpenClaw yalnız solu yapıyor ve bunu açıkça söylüyor:</text><text x="14" y="156" font-size="9" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">"Bir satırın yokluğu hiçbir şey kanıtlamaz."</text><text x="14" y="172" font-size="7.6" fill="#767d84" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">KKB'nin ihtiyacı sağ taraf. Ayrımı baştan yapmak ucuz, sonradan şema göçü.</text></svg>
</div>

<sub>▲ İki kayıt hattı — aynı şey değiller · düzenlemek için: [`f_two_ledgers.excalidraw`](diagrams/wiki/f_two_ledgers.excalidraw) → excalidraw.com'a sürükle</sub>


**Uyum kaydı** ile **hata ayıklama kaydı** aynı şey değildir:

| | Uyum kaydı | Hata ayıklama kaydı |
|---|---|---|
| Değişmez mi | **evet** | hayır |
| Saklama süresi | var | kısa |
| Sır taşır mı | **asla** | taşıyabilir |
| Kim okur | denetçi | mühendis |

Tek hatla ikisini birden yapmaya çalışmak **ikisini de bozar**: ya denetim
kaydına sır sızar, ya hata ayıklama kaydı gereksiz yere ömür boyu saklanır.

---

<a id="s12"></a>
## 12 · Çerçeve seçimi

<div align="center">
<svg viewBox="0 0 600 266" width="600" height="266"><path d="M201,7 L399,7 L399,35 L201,35 Z" fill="#f8f9fa" stroke="none"/><path d="M201.4,4.9 Q300.0,4.5 400.8,7.1 M399.4,5.5 Q399.7,21.0 400.7,37.5 M400.1,35.6 Q300.0,35.4 199.7,36.7 M200.1,36.4 Q199.9,21.0 200.4,4.5" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><path d="M200.7,6.2 Q300.0,5.0 401.1,4.6 M400.0,6.8 Q400.3,21.0 399.0,36.8 M399.4,37.1 Q300.0,36.3 198.7,37.4 M199.6,35.0 Q199.8,21.0 199.9,7.1" fill="none" stroke="#868e96" stroke-width="1.6" stroke-linecap="round"/><text x="300.0" y="24.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">kullanıcı · kurumsal SSO</text><path d="M300.2,37.8 Q300.2,45.0 299.4,50.8" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="3 3"/><path d="M299.7,37.9 Q299.9,45.0 300.3,50.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="3 3"/><text x="410" y="50" font-size="7.4" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">SINIR 1 — kimlik</text><path d="M121,55 L479,55 L479,95 L121,95 Z" fill="#fff4e6" stroke="none"/><path d="M121.3,53.2 Q300.0,55.1 481.0,54.6 M479.3,52.7 Q480.3,75.0 479.7,95.8 M479.9,96.2 Q300.0,96.0 118.8,95.1 M119.9,95.0 Q120.1,75.0 119.3,55.5" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><path d="M120.7,53.5 Q300.0,55.1 479.7,54.7 M480.4,52.5 Q479.9,75.0 480.7,95.0 M480.1,97.4 Q300.0,96.6 119.7,96.5 M121.4,95.2 Q119.5,75.0 120.1,54.6" fill="none" stroke="#e8590c" stroke-width="1.6" stroke-linecap="round"/><text x="300.0" y="74.2" font-size="8.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">KONTROL DÜZLEMİ</text><text x="300.0" y="85.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">kapsam parametreden · rol = tool grubu · onay = donmuş plan</text><path d="M299.8,98.5 Q299.9,105.0 298.4,113.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="3 3"/><path d="M300.3,97.2 Q300.0,105.0 299.9,112.5" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="3 3"/><text x="410" y="110" font-size="7.4" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">SINIR 2 — yetki</text><path d="M121,115 L479,115 L479,153 L121,153 Z" fill="#e7f5ff" stroke="none"/><path d="M118.8,113.0 Q300.0,113.7 481.2,113.1 M480.7,113.1 Q480.0,134.0 481.0,153.9 M479.1,153.8 Q300.0,152.4 119.6,153.3 M119.8,153.7 Q119.9,134.0 121.2,113.4" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><path d="M120.8,113.8 Q300.0,112.4 478.5,114.8 M480.5,115.0 Q479.6,134.0 479.3,155.1 M481.1,155.6 Q300.0,154.5 120.2,153.7 M120.7,154.0 Q120.4,134.0 118.7,113.5" fill="none" stroke="#1971c2" stroke-width="1.6" stroke-linecap="round"/><text x="300.0" y="133.2" font-size="8.6" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">AJAN DÖNGÜSÜ (AutoGen)</text><text x="300.0" y="144.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">yetenek dizini cache sınırının üstünde</text><path d="M218.4,155.7 Q219.9,163.0 221.3,170.7" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="3 3"/><path d="M218.8,156.2 Q220.2,163.0 220.4,169.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="3 3"/><path d="M391.6,156.5 Q390.0,163.0 390.1,168.6" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="3 3"/><path d="M388.5,157.5 Q390.0,163.0 390.1,170.3" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="3 3"/><path d="M121,173 L295,173 L295,209 L121,209 Z" fill="#ebfbee" stroke="none"/><path d="M118.5,173.0 Q208.0,172.7 296.6,170.6 M297.2,173.6 Q295.4,191.0 294.7,211.3 M295.2,208.7 Q208.0,209.0 120.6,208.6 M121.0,209.7 Q120.1,191.0 119.5,171.7" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><path d="M118.6,172.2 Q208.0,170.6 297.0,173.6 M295.4,173.0 Q296.6,191.0 295.3,210.9 M297.1,210.4 Q208.0,208.4 120.7,211.1 M119.5,210.7 Q119.5,191.0 119.9,172.9" fill="none" stroke="#2f9e44" stroke-width="1.6" stroke-linecap="round"/><text x="208.0" y="190.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">tool / API</text><text x="208.0" y="201.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">sandbox</text><path d="M313,173 L487,173 L487,209 L313,209 Z" fill="#fff5f5" stroke="none"/><path d="M312.7,172.3 Q400.0,173.7 488.7,173.4 M486.6,172.9 Q488.6,191.0 489.1,210.5 M487.3,211.5 Q400.0,210.2 312.7,211.6 M313.1,209.1 Q312.2,191.0 312.7,170.7" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><path d="M311.8,172.1 Q400.0,171.4 489.6,172.4 M489.0,172.8 Q488.6,191.0 489.1,211.4 M488.1,210.5 Q400.0,209.0 310.6,209.6 M312.8,208.6 Q311.9,191.0 311.1,171.2" fill="none" stroke="#c92a2a" stroke-width="1.6" stroke-linecap="round"/><text x="400.0" y="190.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">dış içerik</text><text x="400.0" y="201.0" font-size="6.8" fill="#767d84" font-family="DejaVu Sans Mono, monospace" text-anchor="middle">sarmalayıcı</text><text x="14" y="192" font-size="7.4" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">SINIR 3</text><text x="500" y="192" font-size="7.4" fill="#c92a2a" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif">SINIR 4</text><path d="M121,225 L479,225 L479,257 L121,257 Z" fill="#f8f0fc" stroke="none"/><path d="M119.9,225.6 Q300.0,222.9 479.5,222.6 M480.8,223.7 Q480.7,241.0 481.6,257.2 M479.4,259.5 Q300.0,256.3 121.4,257.7 M120.4,258.3 Q119.4,241.0 120.4,222.9" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><path d="M119.5,223.4 Q300.0,225.2 478.6,222.5 M478.8,223.4 Q480.1,241.0 480.8,258.8 M479.1,258.4 Q300.0,258.2 119.6,259.6 M121.1,257.3 Q120.4,241.0 120.9,223.8" fill="none" stroke="#5f3dc4" stroke-width="1.6" stroke-linecap="round"/><text x="300.0" y="244.2" font-size="8" fill="#1e1e1e" font-family="Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif" font-weight="bold" text-anchor="middle">İKİ KAYIT HATTI + telemetri (içerik yok, boyut var)</text><path d="M298.8,212.0 Q300.1,217.0 300.5,221.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="3 3"/><path d="M300.3,212.1 Q300.2,217.0 300.7,223.2" fill="none" stroke="#1e1e1e" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="3 3"/></svg>
</div>

<sub>▲ Üç ayrı ilişki · düzenlemek için: [`f_atlas.excalidraw`](diagrams/wiki/f_atlas.excalidraw) → excalidraw.com'a sürükle</sub>


### Bakım modu bir söylenti değil — ölçüldü

| Paket | Son sürüm | Kaç gün önce |
|---|---|---:|
| **autogen-agentchat** | 0.7.5 | **323** |
| semantic-kernel | 1.44.1 | 13 |
| langgraph | 1.2.11 | 8 |
| agent-framework (MAF) | 1.14.0 | 5 |
| crewai | 1.15.16 | 5 |
| google-adk | 2.7.1 | 2 |
| openai-agents | 0.22.0 | **0** |

Rakiplerin hepsi son iki hafta içinde sürüm çıkardı; AutoGen on bir ay önce.

### Ama MAF'a bugün geçmek de bedava değil

* GA'dan sonra **iki ayda 15 kırıcı değişiklik** — Microsoft'un kendi
  işaretlemesiyle **[kaynak]**
* 36 paketin **8'i** kararlı; harness, FIDES, beceriler hepsi `experimental`
* **Dağıtık runtime yok** — ve LangGraph, CrewAI, Agents SDK, ADK'da da yok

### Kararın dayanağı: motor değiştirilebilir

54 modülün **17'si** AutoGen içe aktarıyor. Kodun **%72,5'i** altında hangi
motorun döndüğünü bilmiyor **[ölçüldü]**. Ekrandaki MAF düğmesi bunun kanıtı.

> **Üç ayrı ilişki:** AutoGen'i **gömüyoruz** (motor, ince arayüz arkasında) ·
> OpenClaw'ı **öğreniyoruz** (karar kuralları, kodu değil) · OpenClaw'ı
> mühendislikte **kullanmaya devam ediyoruz**.

---

<a id="s13"></a>
## 13 · Bilinen sınırlar

Bu wiki'nin en önemli bölümü. Her sayının ölçüldüğünü söyleyen bir belge,
ölçmediklerini de sayabilmeli.

| Ne | Durum | Neden |
|---|---|---|
| Kod konteynerinin ağ izolasyonu | **bilinen açık** | Yukarı akış parametre sunmuyor. Konteyner izole, ama ağı var. |
| Prompt enjeksiyonu | **izlenmiyor** | Kapı tool adına ve imzasına bakıyor, verinin nereden geldiğine değil. Tarama sonucuna gömülü talimat kapıdan geçer. |
| Zamanlayıcı | **devredilmiş** | Yerli karşılığı yazıldı ve testli, bağlanmadı. |
| MAF kipi | **dar** | Beş API yüzeyi. Kıyas yüzeyi, ikinci boru hattı değil. Tool çağrılan turda cevap metni boş dönüyor. |
| LangGraph / CrewAI davranışı | **[teyitsiz]** | Kuruldular, sembolleri tarandı, **koşturulmadılar**. "Var" demek "çalışıyor" demek değil. |
| Lobster (OpenClaw eklentisi) | **[teyitsiz]** | Resmî eklenti, çekirdekte değil, kurmadık. |

---

<sub>Üretim: `python docs/tools/make_wiki.py` · şemalar `docs/diagrams/figures.py`
(desteyle aynı çizimler) · düzenlenebilir kaynaklar `docs/diagrams/wiki/`</sub>
