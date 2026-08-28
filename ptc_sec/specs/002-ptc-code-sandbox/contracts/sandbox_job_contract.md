# Contract: Sandbox Job — Giriş/Çıkış

## Giriş (ana asistan → sandbox)

- **Kod**: bir `ConfigMap` (`ptc-code-{run_id}`) içindeki `code.py` anahtarı,
  Job'un pod şablonunda `/sandbox/code.py` olarak volume-mount edilir.
- **Tool Gateway adresi**: `TOOL_GATEWAY_ENDPOINT` ortam değişkeni (Service
  ClusterIP:port) — Job oluşturulurken ana asistan tarafından enjekte edilir.
- **Zaman aşımı**: Job'un `spec.activeDeadlineSeconds` alanı (varsayılan: 30).

## Çıkış (sandbox → ana asistan)

- Sandbox'ın `entrypoint.py`'si, `code.py`'yi çalıştırır ve **nihai sonucu
  stdout'a tek bir JSON satırı olarak** yazar:

  ```json
  {"status": "success", "result": "..."}
  {"status": "error", "message": "..."}
  ```

- **Tool-call satırları (T015 uzantısı)**: her Tool Gateway çağrısından önce,
  `entrypoint.py` ayrıca ayrı bir JSON satırı yazar — `"type": "tool_call"` alanı
  bunu nihai satırdan ayırt eder (nihai satırda `type` alanı YOK):

  ```json
  {"type": "tool_call", "tool": "list_open_tickets", "args": {}, "status": "success", "timestamp": "..."}
  ```

  Ana asistan (`sandbox_runner.py`), pod log'undaki HER satırı parse eder:
  `type == "tool_call"` olanları `LiveToolCall`'a çevirip `Trace`'e besler
  (FR-008); `type` alanı olmayan (tek) satır nihai sonuçtur.
- Ana asistan, Job tamamlandıktan sonra pod'un log'unu
  (`read_namespaced_pod_log`, **`_preload_content=False` ile** — bkz. not aşağıda)
  okuyup bu satırları parse eder.

**Not (bulunan bir client tuhaflığı, 2026-08-28)**: `kubernetes` Python
client'ının `read_namespaced_pod_log`'u, log JSON'a benziyorsa onu sessizce
parse edip `str(dict)` ile geri yazıyor — bu da tırnak işaretlerini bozup
JSON'u geçersiz kılıyor. `_preload_content=False` ile ham HTTP yanıtı
okunarak bu atlatılır.
- Job `activeDeadlineSeconds`'ı aşarsa Kubernetes Job'u kendisi sonlandırır
  (`status.conditions[].reason == "DeadlineExceeded"`) — ana asistan bunu
  `status: timeout` olarak yorumlar, pod log'u eksik/yarım olsa bile.

## Exit code sözleşmesi

| Kod | Anlam |
|---|---|
| 0 | `code.py` hatasız bitti (stdout'taki JSON `status: success` veya kendi ele aldığı bir `error` olabilir) |
| ≠0 | `entrypoint.py`'nin kendisi çöktü (ör. `code.py` sözdizim hatası) — ana asistan `status: error` sayar |
