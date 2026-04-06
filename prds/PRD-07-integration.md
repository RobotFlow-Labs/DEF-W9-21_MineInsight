# PRD-07: Integration (Docker + API + ROS2)

> Module: DEF-mineinsight | Priority: P1
> Depends on: PRD-02, PRD-06
> Status: COMPLETE

## Objective
Provide Docker serving infrastructure, FastAPI endpoint, and ROS2 node stubs for
ANIMA ecosystem integration.

## Components
1. **Dockerfile.serve**: 3-layer build from anima-serve:jazzy base
2. **docker-compose.serve.yml**: Profiles for serve, ros2, api, test
3. **FastAPI endpoints**: /health, /ready, /info, /predict
4. **ROS2 node stub**: AnimaNode subclass for mineinsight detection

## Acceptance Criteria
- [x] Dockerfile.serve builds from anima-serve:jazzy.
- [x] docker-compose.serve.yml has all required profiles.
- [x] /health and /predict endpoints defined.
- [x] ROS2 topic configuration in anima_module.yaml.

## Files Created
| File | Purpose | Est. Lines |
|---|---|---:|
| `Dockerfile.serve` | 3-layer Docker build | ~40 |
| `docker-compose.serve.yml` | Compose with profiles | ~60 |

## Test Plan
```bash
# Build and test health endpoint
docker compose -f docker-compose.serve.yml --profile api build
docker compose -f docker-compose.serve.yml --profile api up -d
curl http://localhost:8080/health
```

## References
- ANIMA Docker Serving standards
- anima-serve base: /mnt/forge-data/docker-base/anima-serve/
