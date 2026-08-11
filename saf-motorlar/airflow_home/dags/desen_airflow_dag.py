"""
desen_airflow_dag.py — Üç deseni Airflow'da kur.

AIRFLOW'UN WORKFLOW'A BAKIŞI: workflow = DOSYA (statik graf tanımı).
Scheduler bu dosyayı PARSE eder ve düğüm kümesini oradan öğrenir. Dolayısıyla
"çalışırken karar ver" ve "çalışırken N tane üret" özel kavramlar gerektirir:

  D2 zincir  : `a >> b >> c` — en doğal olduğu yer.
  D3 koşullu : BranchPythonOperator — DÜĞÜM ADI döndürür, seçilmeyen dal
               `skipped` durumuna geçer. Sonrasında birleşmek istersen
               `trigger_rule="none_failed_min_one_success"` GEREKİR, yoksa
               alt düğüm de skip olur. Yani dal + birleşme iki ayrı kavram.
  D4 dinamik : Dynamic Task Mapping (Airflow 2.3+) — `.expand()`. Düğüm
               SAYISI çalışma anında belli olur ama düğüm TÜRÜ dosyada sabit.
               "Yeni bir iş türü ekle" hâlâ imkânsız.

Bu üç desende Airflow'un karakteri net görünüyor: her şey bir düğümdür ve
her düğümün bir DURUMU vardır (success/failed/skipped/upstream_failed) —
diğer motorlarda "atlandı" diye bir kayıt yok, dal hiç var olmuyor.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

default_args = {"retries": 2, "retry_delay": timedelta(seconds=1)}


# ═════════ D2 · ZİNCİR ═════════
def _z(n, **ctx):
    import desenler as D
    onceki = ctx["ti"].xcom_pull(task_ids=f"z{n-1}") if n > 1 else None
    return D.zincir_adim(n, onceki)


# ═════════ D3 · KOŞULLU ═════════
def _tarama(**ctx):
    import desenler as D
    return D.tarama()


def _dal_sec(**ctx):
    """BranchPythonOperator: koşacak düğümün ADINI döndürür.
    Döndürülmeyen dal `skipped` olur — Airflow'a özgü, AÇIK bir durum."""
    t = ctx["ti"].xcom_pull(task_ids="tarama")
    return "dal_raporla" if t["asti"] else "dal_atla"


def _raporla(**ctx):
    import desenler as D
    return D.dal_raporla(ctx["ti"].xcom_pull(task_ids="tarama"))


def _atla(**ctx):
    import desenler as D
    return D.dal_atla(ctx["ti"].xcom_pull(task_ids="tarama"))


def _birlesim(**ctx):
    """Dallar birleşiyor. trigger_rule olmadan bu düğüm de SKIP olurdu."""
    ti = ctx["ti"]
    for d in ("dal_raporla", "dal_atla"):
        v = ti.xcom_pull(task_ids=d)
        if v:
            return v
    return {"dal": "?", "mesaj": "hiçbir dal sonuç vermedi"}


# ═════════ D4 · DİNAMİK ═════════
def _bul(**ctx):
    import desenler as D
    return D.dosyalari_bul()["dosyalar"]      # expand() LİSTE bekler


def _isle(path):
    import desenler as D
    return D.dosya_isle(path)


def _topla(**ctx):
    import desenler as D
    return D.topla(ctx["ti"].xcom_pull(task_ids="isle"))


with DAG(dag_id="desen_zincir", default_args=default_args, schedule=None,
         start_date=datetime(2026, 1, 1), catchup=False, tags=["desen"]) as dag_z:
    onceki = None
    for i in range(1, 11):
        t = PythonOperator(task_id=f"z{i}", python_callable=_z, op_kwargs={"n": i})
        if onceki:
            onceki >> t
        onceki = t


with DAG(dag_id="desen_kosullu", default_args=default_args, schedule=None,
         start_date=datetime(2026, 1, 1), catchup=False, tags=["desen"]) as dag_k:
    tarama = PythonOperator(task_id="tarama", python_callable=_tarama)
    sec = BranchPythonOperator(task_id="dal_sec", python_callable=_dal_sec)
    rap = PythonOperator(task_id="dal_raporla", python_callable=_raporla)
    atl = PythonOperator(task_id="dal_atla", python_callable=_atla)
    # trigger_rule ŞART: varsayılan all_success olsaydı, skip olan dal yüzünden
    # birleşim düğümü de skip olurdu. Dal + birleşme iki ayrı kavram.
    bir = PythonOperator(task_id="birlesim", python_callable=_birlesim,
                         trigger_rule="none_failed_min_one_success")
    tarama >> sec >> [rap, atl] >> bir


with DAG(dag_id="desen_dinamik", default_args=default_args, schedule=None,
         start_date=datetime(2026, 1, 1), catchup=False, tags=["desen"]) as dag_d:
    bul = PythonOperator(task_id="bul", python_callable=_bul)
    # Dynamic Task Mapping: düğüm SAYISI çalışma anında, TÜRÜ dosyada sabit.
    isle = PythonOperator.partial(task_id="isle", python_callable=_isle) \
        .expand(op_args=bul.output.map(lambda p: [p]))
    topla = PythonOperator(task_id="topla", python_callable=_topla)
    isle >> topla
