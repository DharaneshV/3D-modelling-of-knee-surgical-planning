# KneeTwin — Knee Surgical Planning Twin

Turns a knee MRI (or CT) into patient-specific 3D bone and cartilage geometry,
measures it, plans a TKA resection, fits generic parametric implant components
to the result, and serves all of it — including a 1:1-scale AR model — from a
single web app.

> [!WARNING]
> Research/POC software. Not a medical device and not validated for clinical
> use. Read [MODEL_CARD.md](MODEL_CARD.md) before interpreting any number it
> produces — several outputs carry real caveats (no mechanical axis is
> recoverable from a knee-only field of view, cartilage loss is systematically
> under-reported, implant components are generic rather than any real product).

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

For running the test suite as well:

```bash
pip install -r requirements-dev.txt
```

Segmentation additionally needs external models that are **not** pip
dependencies: TotalSegmentator (CT track, GPU) and CartiMorph (MRI track, run
via WSL against `~/cartimorph_venv`). The viewer, resection planning, implant
fitting and AR export all work against already-processed cases without them.

## Running

One process serves both the API and the frontend:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000>.

`--host 0.0.0.0` makes it reachable from other devices on the LAN, which is
what phone AR testing needs. Drop it (or use `127.0.0.1`) to keep it local —
see the security note below before exposing it.

### Opening an already-processed case

Uploading a scan is the normal route, but a processed case can be opened
directly:

```
http://localhost:8000/?task=<task_id>
```

Task IDs are the directory names under `meshes/`. This is how to reach the
dashboard from a phone, where uploading a `.nii.gz` isn't practical.

### AR on a phone

The **View in AR** button appears once a task has an AR model. Camera AR needs
an ARCore-capable Android device **and HTTPS** — WebXR refuses to start
otherwise, and `localhost` is the only exempt origin, so a plain LAN IP will
load the page but not offer AR. A tunnel is the simplest way to get a trusted
certificate:

```bash
cloudflared tunnel --url http://localhost:8000
```

That URL is public while it runs. See the security note before using it with
real data.

## Tests

```bash
pytest tests/ -q
```

85 tests, no GPU or external models required — they run against synthetic
fixtures rather than real cases.

## Maintenance

Nothing expires on its own; `uploads/` and `meshes/` grow with every upload.

```bash
python scripts/cleanup_old_tasks.py --days 30            # dry run, prints what would go
python scripts/cleanup_old_tasks.py --days 30 --delete   # actually remove
```

Segmentation accuracy can be re-validated against the OAI-ZIB test set with
`python scripts/run_oaizib_validation.py`, which writes
`results/dice_scores_summary.csv`.

## Security

There is **no authentication of any kind**. Any caller who can reach the port
can upload scans and read every task's meshes, reports and PDFs. Uploads are
capped at 500MB and filenames/task IDs are validated against path traversal,
but that is containment, not access control. Do not expose this to an untrusted
network — or leave a tunnel running — with real patient data on it.

## Layout

| Path | What's in it |
|---|---|
| `backend/` | FastAPI app (`main.py`), pipeline orchestration, PDF report generation |
| `frontend/` | Single-page UI — 2D slice viewer, Three.js 3D viewer, AR modal. Served by the backend |
| `src/segmentation/` | CT (TotalSegmentator) and MRI (CartiMorph) segmentation, accuracy metrics |
| `src/mesh/` | SurfaceNets meshing, topology repair, resection planes, implant geometry, AR glTF export |
| `src/measurements/` | Clinical measurements (JSW, sizing) |
| `scripts/` | Batch processing, validation, maintenance |
| `tests/` | pytest suite |
| `meshes/`, `uploads/`, `tasks/`, `cache/` | Generated per-task output (gitignored) |

## Further reading

- [MODEL_CARD.md](MODEL_CARD.md) — accuracy figures, failure modes, and the
  caveats that govern how outputs may be read. Start here.
- [Knee_Twin_Implementation_Plan.md](Knee_Twin_Implementation_Plan.md) —
  current architecture and what is/isn't built.
- [docs/Clinical_Validation_Overview.md](docs/Clinical_Validation_Overview.md)
  — validation approach.
