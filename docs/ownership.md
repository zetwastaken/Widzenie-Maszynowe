# Ownership Map

## Module Ownership

- Person 1: `decoder.py`
- Person 2: `sharpness_metric.py` and `exposure_metric.py`
- Person 3: `face_metric.py`
- Person 4: `scorer.py`
- Person 5: `reporter.py`, `main.py`, `docs/*`

## Shared Files (Cross-Team)

- `docs/contracts.md`

## Coordination Policy

1. Feature teams can modify only their owned modules in regular tasks.
2. Any change in scoring keys or class weights/thresholds must be announced before merge.
3. Data-flow changes require updating `docs/contracts.md` in the same PR.
4. Integration order follows pipeline stages:
   decoder -> metrics -> scoring -> reporting -> main
