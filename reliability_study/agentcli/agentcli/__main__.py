"""`python3 -m agentcli` → uygulamayı açar."""
import argparse
import sys

from .shell import Uygulama


def main() -> int:
    p = argparse.ArgumentParser(prog="agentcli")
    p.add_argument("--workdir", default=None, help="terminal calisma dizini")
    p.add_argument("--port", type=int, default=9222, help="Chrome CDP portu")
    p.add_argument("--strategy", default=None, help="baslangic zihniyeti")
    p.add_argument("--tema-kapali", action="store_true",
                   help="terminalin arka planini beyaza cevirme")
    a = p.parse_args()
    if a.tema_kapali:
        import os
        os.environ["AGENTCLI_TEMA"] = "kapali"
    uyg = Uygulama(port=a.port, workdir=a.workdir)
    if a.strategy:
        uyg.strateji = a.strategy
    try:
        return uyg.calistir()
    except SystemExit as e:
        return int(e.code or 0)


if __name__ == "__main__":
    sys.exit(main())
