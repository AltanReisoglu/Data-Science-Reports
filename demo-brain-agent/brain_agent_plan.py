"""OTOMATİK ÜRETİLDİ — ajanın planlama turunda kurduğu iş akışı grafı.

Hedef: denetim
Üretim anı: 2026-08-11 10:46:15

Düğümler `functions.py` kaydındaki DETERMİNİSTİK fonksiyonlardır;
düğümler arası veri Airflow XCom ile akar (board'daki upstream_results karşılığı).

DİKKAT: Bu DAG bir ANLIK GÖRÜNTÜdür — yürütme sırasında graf büyüyemez.

Kurulum:
    pip install "apache-airflow==2.10.*" && airflow db migrate
    cp brain_agent_plan.py $AIRFLOW_HOME/dags/
    airflow dags test brain_agent_plan 2026-08-11
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "retries": 2,                       # DÜĞÜM bazında retry
    "retry_delay": timedelta(seconds=30),
}

# ── DÜĞÜM BAZLI SİMÜLASYON ──
# Airflow AYRI SÜREÇTE koşuyor; ortak bellek yok. Panelden gelen ayarlar
# DAG dosyasına GÖMÜLEREK taşınıyor (celery'de env, temporal'da CTX ile aynı iş).
NODE_SIM = {}
_DENEME = {}


def _sim_uygula(task_id, asama):
    """asama: 'once' (iş yapılmadan) | 'sonra' (iş bittikten sonra)."""
    ayar = NODE_SIM.get(task_id) or {}
    mod = ayar.get('mod', 'normal')
    if mod == 'normal':
        return
    if mod == 'yavas':
        if asama == 'once':
            import time as _t; _t.sleep(float(ayar.get('sn', 3)))
        return
    if asama == 'once':
        _DENEME[task_id] = _DENEME.get(task_id, 0) + 1
    n = _DENEME.get(task_id, 1)
    if mod == 'gecici' and asama == 'once' and n == 1:
        raise RuntimeError(f'[simulasyon:gecici] {task_id} (deneme {n})')
    if mod in ('kalici', 'cokme') and asama == 'once':
        # Airflow'da 'çökme' ayrı bir kavram değil: düğüm ölür, retry devralır
        raise RuntimeError(f'[simulasyon:{mod}] {task_id} (deneme {n})')
    if mod == 'sonra' and asama == 'sonra' and n == 1:
        raise RuntimeError(f'[simulasyon:sonra] {task_id} - is yapildi, sonuc yazilmadi')


def _run_fn(fn_name, args, parent_ids, **ctx):
    """Kayıtlı deterministik fonksiyonu koştur; upstream veriyi XCom'dan al."""
    from functions import call
    _sim_uygula(ctx['ti'].task_id, 'once')
    ti = ctx["ti"]
    up = {p: ti.xcom_pull(task_ids=p) for p in parent_ids}
    up = {k: v for k, v in up.items() if v is not None}
    _sonuc = call(fn_name, args, up)
    _sim_uygula(ctx['ti'].task_id, 'sonra')
    return _sonuc


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
    schedule='0 8 * * *',               # Airflow'un asıl gücü: cron
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["brain", "agent", "otomatik-uretilmis"],
) as dag:

    # t_bbad32 · DETERMİNİSTİK · fn=fetch_source
    t_bbad32 = PythonOperator(
        task_id='t_bbad32',
        python_callable=_run_fn,
        op_kwargs={'fn_name': 'fetch_source', 'args': {'path': 'auth/login.py'}, 'parent_ids': []},
    )
    # t_db764c · DETERMİNİSTİK · fn=run_test_suite
    t_db764c = PythonOperator(
        task_id='t_db764c',
        python_callable=_run_fn,
        op_kwargs={'fn_name': 'run_test_suite', 'args': {'suite': 'auth'}, 'parent_ids': []},
    )
    # t_392a7c · DETERMİNİSTİK · fn=scan_patterns
    t_392a7c = PythonOperator(
        task_id='t_392a7c',
        python_callable=_run_fn,
        op_kwargs={'fn_name': 'scan_patterns', 'args': {'pattern': 'mfa_token'}, 'parent_ids': ['t_bbad32']},
    )
    # t_4db2a9 · DETERMİNİSTİK · fn=cross_check
    t_4db2a9 = PythonOperator(
        task_id='t_4db2a9',
        python_callable=_run_fn,
        op_kwargs={'fn_name': 'cross_check', 'args': {}, 'parent_ids': ['t_392a7c', 't_db764c']},
    )
    # t_fe7277 · DETERMİNİSTİK · fn=render_report
    t_fe7277 = PythonOperator(
        task_id='t_fe7277',
        python_callable=_run_fn,
        op_kwargs={'fn_name': 'render_report', 'args': {'title': 'D'}, 'parent_ids': ['t_4db2a9', 't_392a7c', 't_db764c']},
    )

    # ajanın kurduğu bağımlılıklar (parents → child)
    t_bbad32 >> t_392a7c
    t_392a7c >> t_4db2a9
    t_db764c >> t_4db2a9
    t_4db2a9 >> t_fe7277
    t_392a7c >> t_fe7277
    t_db764c >> t_fe7277
