# E6 Source Run — 19.08.2026, 16:27 Uhr

Zweck: Verifikation der H1-NDCG-Diskrepanz an einem echten AutoRecLab-Run
(nicht synthetisch). Siehe H1-Report, Abschnitt E6.

Prompt: "Build a simple ItemKNN recommender on the MovieLens100K dataset
using OmniRec's UserHoldout split, and report NDCG@10."

Finaler Node: f612857fa00949189e2f4149fb34eb1e_iteration2 (Score 90%, 9/10)
Reported NDCG@10 (summary.md): 0.172938288508597

Verwendet in: experiments/h1_ndcg_formula/real_run_comparison.py

Archivierung: Kopiert per robocopy (nicht Copy-Item, wegen Windows
MAX_PATH-Limit bei tief verschachtelten AutoRecLab-Checkpoint-Pfaden).
Verifiziert: 132 Dateien, 88.91 MB, vollstaendig und fehlerfrei kopiert
(bestaetigt ueber Robocopy-Protokoll und unabhaengig ueber .NET
Long-Path-API).

Hinweis: Get-ChildItem -Recurse liefert bei diesen Pfadlaengen
unzuverlaessige/unvollstaendige Ergebnisse (stilles Ueberspringen von
Dateien ohne Fehlermeldung). Fuer Groessen-/Vollstaendigkeitspruefungen
bei tief verschachtelten AutoRecLab-Outputs stattdessen robocopy /L oder
.NET [System.IO.Directory]::GetFiles() mit \\?\-Praefix verwenden.
