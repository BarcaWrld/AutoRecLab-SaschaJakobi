# Smoke Test — 18.08.2026, 21:11 Uhr

Zweck: Verifikation, dass Setup (uv, .env/API-Key, MCP-Verbindung, Embeddings) grundsätzlich funktioniert.
Kein Teil der H1/H2-Hypothesenprüfung.

Prompt: "Build a tiny popularity baseline on a small MovieLens sample and report Recall@10."
Dataset: MovieLensLatestSmall (Agent-Wahl, nicht MovieLens100K wie für H1/H2 geplant)
Ergebnis: Recall@10 = 0.0317 (LensKit.PopScorer)
Endscore: 70% (7/10) nach Refinement, "ended without full satisfaction"

Nebenbeobachtung (nicht verifiziert, nur beiläufig aufgefallen):
Mehrere Draft-/Refinement-Nodes scheiterten am selben Bug-Muster: UserHoldout-Split
schlägt fehl bei zu kleinen/spärlichen Sample-Größen (n_samples=1 nach Subsampling).
Evtl. relevant als Ausgangspunkt für eine eigene Beobachtung im Rahmen von H2 --
noch nicht als Hypothese formuliert, nur roh notiert.
