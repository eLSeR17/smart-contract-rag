# scripts/hooks — pre-push hook

Replica localmente las comprobaciones del CI (ruff + EXE + pytest + pip-audit) antes de cada push. **Instalación**: `git config core.hooksPath scripts/hooks`.

Nota: EXE001/EXE002 van replicados en el hook porque ruff no aplica las reglas EXE en WSL2
(documentado en `ruff rule EXE002`), mientras que el runner de GitHub sí las aplica.
