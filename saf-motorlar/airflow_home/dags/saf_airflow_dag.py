"""
saf_airflow_dag.py — Airflow'un SAF hâli: statik DAG dosyası.

DİKKAT: Bu dosya bir DAG TANIMIDIR, koşturulmaz. Airflow'un çalışma modeli
budur — grafı bir dosyaya yazarsın, scheduler onu PARSE eder, sonra koşturur.

Graf burada VERİ değil, DOSYA. Bunun iki sonucu var:
  • Kalıcılık ve zamanlama bedava gelir (metadata DB + scheduler).
  • Graf PARSE ZAMANINDA sabitlenir; çalışırken değişemez.

Veri akışı XCom ile: her task'ın dönüşü metadata DB'ye yazılır, alt task
`ti.xcom_pull(task_ids=...)` ile çeker. Bizim `upstream_results`ın karşılığı.

Kurulum:
    pip install "apache-airflow==2.10.*" && airflow db migrate
    cp saf_airflow_dag.py $AIRFLOW_HOME/dags/
    airflow dags test saf_denetim 2026-08-10
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "retries": 2,                          # DÜĞÜM bazında retry — Airflow'un kendi
    "retry_delay": timedelta(seconds=30),  # backoff kutudan geliyor
}


def _cek(**ctx):
    import isakisi as J
    return J.cek()


def _test(**ctx):
    import isakisi as J
    return J.test()


def _tara(**ctx):
    import isakisi as J
    # veri akışı: XCom'dan çek
    return J.tara(ctx["ti"].xcom_pull(task_ids="cek"))


def _esle(**ctx):
    import isakisi as J
    ti = ctx["ti"]
    return J.esle(ti.xcom_pull(task_ids="tara"), ti.xcom_pull(task_ids="test"))


def _rapor(**ctx):
    import isakisi as J
    ti = ctx["ti"]
    return J.rapor(ti.xcom_pull(task_ids="esle"),
                   ti.xcom_pull(task_ids="tara"),
                   ti.xcom_pull(task_ids="test"))


with DAG(
    dag_id="saf_denetim",
    default_args=default_args,
    schedule="0 8 * * *",          # Airflow'un asıl gücü: cron kutudan
    start_date=datetime(2026, 1, 1),
    catchup=False,                 # True olsaydı kaçırılan günler telafi edilirdi
    max_active_runs=1,
    tags=["saf", "kiyas"],
) as dag:
    cek = PythonOperator(task_id="cek", python_callable=_cek)
    test = PythonOperator(task_id="test", python_callable=_test)
    tara = PythonOperator(task_id="tara", python_callable=_tara)
    esle = PythonOperator(task_id="esle", python_callable=_esle)
    rapor = PythonOperator(task_id="rapor", python_callable=_rapor)

    # Bağımlılıklar — graf burada, dosyada sabit.
    # trigger_rule varsayılan 'all_success': üst battıysa alt 'upstream_failed' olur
    # (bizim 'cancelled'ımızın karşılığı, kutudan geliyor).
    cek >> tara
    [tara, test] >> esle
    [esle, tara, test] >> rapor
