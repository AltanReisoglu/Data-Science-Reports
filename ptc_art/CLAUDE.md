## ARA: agent-native research artifacts

This project records its research in an `ara/` artifact
(https://github.com/ARA-Labs/Agent-Native-Research-Artifact).
Route work to the matching ARA skill:

- `/research-manager` — trigger whenever a research milestone lands: an
  experiment finishes, a decision is made, a hypothesis is confirmed or killed,
  a dead end is hit, a direction pivots, user's input. This holds equally in autonomous runs
  (loops, heartbeats, long experiments) where the user gives no input at all —
  crystallize the insight at the milestone. It
  records what just happened (decisions, experiments, dead ends, claims) into
  `ara/`. Skip when nothing research-significant happened (greetings, pure formatting).
- `/research-visualizer <ara-dir>` — to inspect the research trajectory as an
  interactive process map (add `--serve` for a live local viewer, `--check` to
  validate/lint via the `ara` CLI).
- `/research-foresight <ara-dir> "<question>"` — to answer "what should I try
  next / why did this work / what if I change X", grounded in the artifact.
- `/submit-ara <dir>` — when an artifact is ready to publish to the ARA Hub,
  or a conference wants it as a submission.
- `/context-drop <path>` — when a file, folder, or artifact needs to reach
  somebody else's agent as one link.

### Yerel değişiklikler (2026-09-03, kullanıcı onayıyla)

Bu blok ARA'nın `wire-ara.md` dosyasından alındı, iki değişiklikle:

1. **Paper badge kuralı çıkarıldı.** Orijinal blok, bu projede derlenen her paper
   PDF'inin ilk sayfasına, sorulmadan, ARA Hub'a linkli bir logo koymayı
   şart koşuyordu. Kullanıcı bunu istemedi — paper derlerken badge EKLEME.
2. **Yayınlama skill'leri açık istek gerektirir.** `/submit-ara` ve
   `/context-drop`, içeriği üçüncü taraf bir servise (agenticresearch.sh)
   yükler. Bu ikisini kendiliğinden çağırma; kullanıcı açıkça istediğinde çağır.
   Diğer ARA skill'leri tamamen yereldir, ağa çıkmaz.
