# Artifact deposu — yerel (MinIO) ve OpenShift (OBC) yolları

**Tek kural:** uygulama kodu bucket bağlantısını YALNIZCA şu ortam
değişkenlerinden okur — bunlar bir `ObjectBucketClaim`'in ürettiği adların
birebir aynısıdır:

| Kaynak | Anahtarlar |
|---|---|
| ConfigMap | `BUCKET_NAME`, `BUCKET_HOST`, `BUCKET_PORT` (+ `BUCKET_REGION`) |
| Secret | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |

Sonuç: **yerel ile OpenShift arasında kod değişmiyor, yalnızca hangi
manifestin uygulandığı değişiyor.**

## Yerel (kind)

    oc/kubectl apply -f k8s/artifact-store/minio.yaml
    oc/kubectl apply -f k8s/artifact-store/obc-shape.local.yaml

`obc-shape.local.yaml`, gerçek bir OBC'nin ürettiği ConfigMap+Secret çiftini
elle taklit eder.

## OpenShift (ODF/NooBaa kuruluysa)

    oc apply -f k8s/artifact-store/objectbucketclaim.yaml

MinIO'ya da `obc-shape.local.yaml`'a da GEREK YOK: OBC hem bucket'ı hem de
kendi adıyla ConfigMap+Secret'ı üretir. Deployment'taki `envFrom` referansları
aynı adı gösterdiği için değişiklik gerekmez.
