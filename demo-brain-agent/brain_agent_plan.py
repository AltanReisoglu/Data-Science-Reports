"""OTOMATİK ÜRETİLDİ — ajanın planlama turunda kurduğu iş akışı grafı.

Hedef: denetim
Üretim anı: 2026-08-10 21:31:51

Düğümler `functions.py` kaydındaki DETERMİNİSTİK fonksiyonlardır;
düğümler arası veri Airflow XCom ile akar (board'daki upstream_results karşılığı).

DİKKAT: Bu DAG bir ANLIK GÖRÜNTÜdür — yürütme sırasında graf büyüyemez.

Kurulum:
    pip install "apache-airflow==2.10.*" && airflow db migrate
    cp brain_agent_plan.py $AIRFLOW_HOME/dags/
    airflow dags test brain_agent_plan 2026-08-10
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "retries": 2,                       # DÜĞÜM bazında retry
    "retry_delay": timedelta(seconds=30),
}


def _run_fn(fn_name, args, parent_ids, **ctx):
    """Kayıtlı deterministik fonksiyonu koştur; upstream veriyi XCom'dan al."""
    from functions import call
    ti = ctx["ti"]
    up = {p: ti.xcom_pull(task_ids=p) for p in parent_ids}
    up = {k: v for k, v in up.items() if v is not None}
    return call(fn_name, args, up)


def _run_agent(title, body, parent_ids, **ctx):
    """LLM düğümü (istisna): işçi ajanı koştur."""
    from agent import BrainAgentJob
    ti = ctx["ti"]
    up = {p: ti.xcom_pull(task_ids=p) for p in parent_ids}
    job = BrainAgentJob(goal=f"{title}\n\nDetay: {body}\n\nUpstream: {up}",
                        strategy="hermes", budget=3000, max_turns=2)
    return job.run_sync()["answer"]


with DAG(
    dag_id="brain_agent_plan",
    default_args=default_args,
    schedule="0 8 * * *",               # Airflow'un asıl gücü: cron
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["brain", "agent", "otomatik-uretilmis"],
) as dag:

    # t_521532 · DETERMİNİSTİK · fn=fetch_source
    t_521532 = PythonOperator(
        task_id='t_521532',
        python_callable=_run_fn,
        op_kwargs={'fn_name': 'fetch_source', 'args': {'path': 'auth/login.py'}, 'parent_ids': []},
    )
    # t_d0ac2f · DETERMİNİSTİK · fn=run_test_suite
    t_d0ac2f = PythonOperator(
        task_id='t_d0ac2f',
        python_callable=_run_fn,
        op_kwargs={'fn_name': 'run_test_suite', 'args': {'suite': 'auth'}, 'parent_ids': []},
    )
    # t_d7357b · DETERMİNİSTİK · fn=scan_patterns
    t_d7357b = PythonOperator(
        task_id='t_d7357b',
        python_callable=_run_fn,
        op_kwargs={'fn_name': 'scan_patterns', 'args': {'pattern': 'mfa_token'}, 'parent_ids': ['t_521532']},
    )
    # t_a921f5 · DETERMİNİSTİK · fn=cross_check
    t_a921f5 = PythonOperator(
        task_id='t_a921f5',
        python_callable=_run_fn,
        op_kwargs={'fn_name': 'cross_check', 'args': {}, 'parent_ids': ['t_d7357b', 't_d0ac2f']},
    )
    # t_affa54 · DETERMİNİSTİK · fn=render_report
    t_affa54 = PythonOperator(
        task_id='t_affa54',
        python_callable=_run_fn,
        op_kwargs={'fn_name': 'render_report', 'args': {'title': 'D'}, 'parent_ids': ['t_a921f5', 't_d7357b', 't_d0ac2f']},
    )

    # ajanın kurduğu bağımlılıklar (parents → child)
    t_521532 >> t_d7357b
    t_d7357b >> t_a921f5
    t_d0ac2f >> t_a921f5
    t_a921f5 >> t_affa54
    t_d7357b >> t_affa54
    t_d0ac2f >> t_affa54
