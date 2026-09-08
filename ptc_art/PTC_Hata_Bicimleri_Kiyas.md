# Hata geri bildirimi biçimleri — ölçülmüş karşılaştırma

`scripts/hata_bicim_kiyas.py` ile üretildi; bulgular
`tests/unit/test_hata_bicimleri.py` ile sabitlendi.

## Yöntem

Aynı beş arıza, on biçimlendiriciye AYNI ham malzemeyle veriliyor
(exception nesnesi, hata anına kadarki stdout, çıkış kodu, süre). Üretilen
metin yedi ölçütte taranıyor, ayrıca tekrar tespiti ve bayt ölçülüyor.

| Senaryo | Ne test ediyor |
|---|---|
| `keyerror` | stdout'tan sonra hata — çıktı korunuyor mu |
| `buyuk` | 160 KB çıktı, sonra hata — kırpma davranışı |
| `ag` | ağ engeli — düzeltilemez hata |
| `syntax` | derleme hatası — kullanıcı karesi YOK |
| `sonucsuz` | exception yok, yalnızca eksik sonuç |

**`Bizim (şimdi)` gerçek koddur** — `sandbox_image/entrypoint.py::_hata_metni`
içe aktarılıp doğrudan çağrılıyor. Diğer dokuzu
`PTC_Error_Recovery_Piyasa_Arastirmasi.md`'de belgelenmiş davranıştan
YENİDEN KURULMUŞTUR; ilgili ürünün kaynak kodu değildir.

## Neyi ölçmüyor — önce bu

Bu tablo **düzeltme başarı oranını ölçmez**. Onun için LLM değerlendirmesi
gerekir (N senaryo x M biçim x k tekrar); burada yok. Buradaki sıralama
"bayt başına taşınan onarım sinyali"dir.

Ayrıca ölçüt listesi bizim araştırmamızdan çıktı ve bizim biçimimizi de o
araştırma şekillendirdi — yani **sıralamada dairesellik var**. Dürüst okuma:
biçimimiz kendi ölçütlerinde iyi çıkıyor; asıl kıyaslanabilir sayı, aynı
sinyali kaç baytla taşıdığımız.

## Sinyal tablosu — 5 senaryonun kaçında sağlanıyor

| Biçim | tip | satır | kaynak | stdout | çıkış | kırpma | talimat | sinyal | tekrar | sızıntı | ort. bayt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Bizim (eski)** | 1/5 | 2/5 | 2/5 | 1/5 | ✗ | 4/5 | ✗ | **10/35** | ✗ | 0 | 56 |
| **Claude Code** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | **30/35** | ✗ | 2 | 2 701 |
| **Codex** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | **30/35** | ✗ | 2 | 2 356 |
| **Anthropic API** | ✓ | ✓ | ✓ | ✓ | ✓ | 4/5 | ✗ | **29/35** | ✗ | 2 | 43 437 |
| **smolagents** | ✓ | 2/5 | ✓ | ✓ | ✗ | ✓ | ✗ | **22/35** | ✗ | 0 | 4 114 |
| **AutoGen / AG2** | ✓ | ✓ | ✓ | ✓ | ✓ | 4/5 | ✗ | **29/35** | ✗ | 2 | 42 587 |
| **SWE-agent (varsayılan)** | 4/5 | 4/5 | 4/5 | ✓ | ✗ | ✓ | 1/5 | **23/35** | ✗ | 2 | 2 331 |
| **SWE-agent (bash_only)** | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | 1/5 | **26/35** | ✗ | 2 | 2 331 |
| **OpenHands** | ✓ | ✓ | ✓ | ✓ | ✓ | 4/5 | ✗ | **29/35** | ✓ | 2 | 42 555 |
| **Bizim (şimdi)** | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | **30/35** | ✗ | 0 | 4 538 |

`sinyal` = 7 ölçüt x 5 senaryo. `tekrar` = aynı arıza üst üste gelince mesaj değişiyor mu. `sızıntı` = mesajda görünen KULLANICI DIŞI traceback karesi (0 iyi; sayının kendisi harness'a bağlı, 0-mı-değil-mi anlamlı).

## Senaryo başına bayt

| Biçim | keyerror | buyuk | ag | syntax | sonucsuz |
|---|---|---|---|---|---|
| Bizim (eski) | 39 | 53 | 80 | 68 | 38 |
| Claude Code | 421 | 12 056 | 697 | 307 | 24 |
| Codex | 476 | 10 113 | 752 | 361 | 78 |
| Anthropic API | 508 | 215 415 | 788 | 389 | 85 |
| smolagents | 96 | 20 173 | 168 | 106 | 28 |
| AutoGen / AG2 | 453 | 211 360 | 729 | 339 | 56 |
| SWE-agent (varsayılan) | 409 | 10 254 | 685 | 295 | 12 |
| SWE-agent (bash_only) | 409 | 10 254 | 685 | 295 | 12 |
| OpenHands | 421 | 211 328 | 697 | 307 | 24 |
| Bizim (şimdi) | 526 | 20 615 | 586 | 469 | 495 |

## Sıralama — sinyal, sonra tekrar, sonra sızıntı, sonra bayt

1. **Bizim (şimdi)** — sinyal 30/35  tekrar yok  sızıntı 0  ort. 4 538 bayt
2. **Codex** — sinyal 30/35  tekrar yok  sızıntı 2  ort. 2 356 bayt
3. **Claude Code** — sinyal 30/35  tekrar yok  sızıntı 2  ort. 2 701 bayt
4. **OpenHands** — sinyal 29/35  tekrar var  sızıntı 2  ort. 42 555 bayt
5. **AutoGen / AG2** — sinyal 29/35  tekrar yok  sızıntı 2  ort. 42 587 bayt
6. **Anthropic API** — sinyal 29/35  tekrar yok  sızıntı 2  ort. 43 437 bayt
7. **SWE-agent (bash_only)** — sinyal 26/35  tekrar yok  sızıntı 2  ort. 2 331 bayt
8. **SWE-agent (varsayılan)** — sinyal 23/35  tekrar yok  sızıntı 2  ort. 2 331 bayt
9. **smolagents** — sinyal 22/35  tekrar yok  sızıntı 0  ort. 4 114 bayt
10. **Bizim (eski)** — sinyal 10/35  tekrar yok  sızıntı 0  ort. 56 bayt

## Modele fiilen giden metin — senaryo `keyerror`

### Bizim (eski)  ·  39 bayt

```
Hata: 'yok'
Tahmini bir değer üretme.
```

### Claude Code  ·  421 bayt

```
Exit code 1
veri yuklendi
Traceback (most recent call last):
  File "/home/altan/Desktop/Data-Science-Reports/ptc_art/scripts/hata_bicim_kiyas.py", line 124, in kodu_kosttur
    exec(compile(kod, yol, "exec"), {"__name__": "__main__"})  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/hata-kiyas-34d4emcs/keyerror.py", line 4, in <module>
    d["yok"]
    ~^^^^^^^
KeyError: 'yok'

```

### Codex  ·  476 bayt

```
Exit code: 1
Wall time: 0.4 seconds
Total output lines: 10
Output:
veri yuklendi
Traceback (most recent call last):
  File "/home/altan/Desktop/Data-Science-Reports/ptc_art/scripts/hata_bicim_kiyas.py", line 124, in kodu_kosttur
    exec(compile(kod, yol, "exec"), {"__name__": "__main__"})  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/hata-kiyas-34d4emcs/keyerror.py", line 4, in <module>
    d["yok"]
    ~^^^^^^^
KeyError: 'yok'

```

### Anthropic API  ·  508 bayt

```
{
 "stdout": "veri yuklendi\n",
 "stderr": "Traceback (most recent call last):\n  File \"/home/altan/Desktop/Data-Science-Reports/ptc_art/scripts/hata_bicim_kiyas.py\", line 124, in kodu_kosttur\n    exec(compile(kod, yol, \"exec\"), {\"__name__\": \"__main__\"})  # noqa: S102\n    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/tmp/hata-kiyas-34d4emcs/keyerror.py\", line 4, in <module>\n    d[\"yok\"]\n    ~^^^^^^^\nKeyError: 'yok'\n",
 "return_code": 1,
 "error_code": "KeyError"
}
```

### smolagents  ·  96 bayt

```
Code execution failed at line 'd["yok"]' due to:
KeyError: 'yok'

Execution logs:
veri yuklendi

```

### AutoGen / AG2  ·  453 bayt

```
exitcode: 1 (execution failed)
Code output:
veri yuklendi
Traceback (most recent call last):
  File "/home/altan/Desktop/Data-Science-Reports/ptc_art/scripts/hata_bicim_kiyas.py", line 124, in kodu_kosttur
    exec(compile(kod, yol, "exec"), {"__name__": "__main__"})  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/hata-kiyas-34d4emcs/keyerror.py", line 4, in <module>
    d["yok"]
    ~^^^^^^^
KeyError: 'yok'

```

### SWE-agent (varsayılan)  ·  409 bayt

```
veri yuklendi
Traceback (most recent call last):
  File "/home/altan/Desktop/Data-Science-Reports/ptc_art/scripts/hata_bicim_kiyas.py", line 124, in kodu_kosttur
    exec(compile(kod, yol, "exec"), {"__name__": "__main__"})  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/hata-kiyas-34d4emcs/keyerror.py", line 4, in <module>
    d["yok"]
    ~^^^^^^^
KeyError: 'yok'

```

### SWE-agent (bash_only)  ·  409 bayt

```
veri yuklendi
Traceback (most recent call last):
  File "/home/altan/Desktop/Data-Science-Reports/ptc_art/scripts/hata_bicim_kiyas.py", line 124, in kodu_kosttur
    exec(compile(kod, yol, "exec"), {"__name__": "__main__"})  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/hata-kiyas-34d4emcs/keyerror.py", line 4, in <module>
    d["yok"]
    ~^^^^^^^
KeyError: 'yok'

```

### OpenHands  ·  421 bayt

```
exitcode: 1
veri yuklendi
Traceback (most recent call last):
  File "/home/altan/Desktop/Data-Science-Reports/ptc_art/scripts/hata_bicim_kiyas.py", line 124, in kodu_kosttur
    exec(compile(kod, yol, "exec"), {"__name__": "__main__"})  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/hata-kiyas-34d4emcs/keyerror.py", line 4, in <module>
    d["yok"]
    ~^^^^^^^
KeyError: 'yok'

```

### Bizim (şimdi)  ·  526 bayt

```
  File "/tmp/hata-kiyas-34d4emcs/keyerror.py", line 4, in <module>
    d["yok"]
    ~^^^^^^^
KeyError: 'yok'

Hata anına kadar yazılan çıktı:
veri yuklendi

Hatayı düzeltip kodu TEKRAR çalıştır. Aynı kodu aynen tekrar gönderme — aynı hatayı verir. Aynı hatayı iki kez aldıysan farklı bir yaklaşım dene. Çok çıktı basıyorsan basmak yerine `/output` altına dosya olarak yaz; dosyalar kalıcı, `print` çıktısı kırpılıyor. Verileri UYDURMA — sonucu ancak kod gerçekten çalışınca bildir.
```
