# Zaman aşımı testi (T024, quickstart.md Senaryo 3 / US3). set_result() hiç
# çağrılmaz — Job'un activeDeadlineSeconds'ı (30s, research.md §6) dolup
# Kubernetes'in kendisi Job'u sonlandırmasını bekliyoruz (FR-007).
while True:
    pass
