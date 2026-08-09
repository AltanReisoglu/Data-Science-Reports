"""
GERÇEK Airflow DAG'ı — zamanlı (scheduled) + retry'lı task-management örneği.

Bu dosya gerçek bir Airflow kurulumunda AYNEN çalışır. Burada ÇALIŞTIRILMADI
(Airflow kurulumu ağır: `pip install apache-airflow` + `airflow db migrate`);
diğer üç POC (Hermes/Temporal/Celery) canlı koşturuldu. Airflow'un ayırt edici
yanları — cron scheduling, catchup/backfill, ve "sadece hatalı düğüm retry olur,
upstream düğümler done kalır" (statik-DAG kısmi resume) — bu DAG'da somut.

Çalıştırmak için (gerçek Airflow'da):
    pip install "apache-airflow==2.10.*"
    export AIRFLOW_HOME=~/airflow && airflow db migrate
    cp airflow_dag.py $AIRFLOW_HOME/dags/
    airflow dags test daily_sales 2026-08-09      # tek koşuyu uçtan uca test et
"""
from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException


def fetch_data(**ctx) -> str:
    ti = ctx["ti"]
    print(f"[fetch_data] try_number={ti.try_number} → veri çekiliyor")
    return f"order:{ctx['ds']}:veri"


def process(**ctx) -> str:
    ti = ctx["ti"]
    print(f"[process] try_number={ti.try_number}")
    if ti.try_number == 1:
        # İLK deneme: geçici hata → Airflow BU DÜĞÜMÜ retry eder (retries=2).
        # Önemli: fetch_data düğümü YENİDEN KOŞMAZ — 'done' kalır (statik-DAG resume).
        raise AirflowException("geçici hata (ödeme timeout) — Airflow bu task'ı retry edecek")
    up = ti.xcom_pull(task_ids="fetch_data")
    return f"{up}|işlendi"


def deliver(**ctx) -> str:
    up = ctx["ti"].xcom_pull(task_ids="process")
    print(f"[deliver] {up} → Slack'e gönderildi")
    return f"{up}|teslim"


with DAG(
    dag_id="daily_sales",
    schedule="0 8 * * *",                                   # cron: her gün 08:00
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Istanbul"),
    catchup=False,                                          # True olsaydı geçmişi BACKFILL ederdi
    max_active_runs=1,                                      # çakışma kontrolü
    default_args={
        "retries": 2,                                       # task-seviyesi retry
        "retry_delay": pendulum.duration(seconds=30),
    },
    tags=["poc", "task-management"],
) as dag:
    t1 = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
    t2 = PythonOperator(task_id="process", python_callable=process)
    t3 = PythonOperator(task_id="deliver", python_callable=deliver)

    t1 >> t2 >> t3     # statik DAG: fetch → process → deliver (şekil önceden belli)
