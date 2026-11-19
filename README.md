# scsp-solver

Set covering / set partitioning 전용 MIP 솔버(C++17 코어 + pybind11). 공개 문헌만 보고 다시 구현하는
클린룸 프로젝트다. 설계와 단계는 [docs/design.md](docs/design.md), 결과는 [docs/p0_report.md](docs/p0_report.md)(하네스),
[docs/p1_report.md](docs/p1_report.md)(SC 휴리스틱), [docs/p2_report.md](docs/p2_report.md)(LP 엔진),
[docs/p3_report.md](docs/p3_report.md)(정확해법), [docs/p4_report.md](docs/p4_report.md)(B&C 강화),
[docs/p5_report.md](docs/p5_report.md)(병렬 B&C·최종 리포트), 기법별 출처는 [docs/references.md](docs/references.md).

**현재 상태: P0~P5 전 단계 구현 완료.** `scsp.solve`는 두 경로로 푼다.

- 커버 행(≥ 1)과 `≤` 부가 제약(계수 ≥ 0)으로 된 모델: Lagrangian 휴리스틱(3스레드) + 루트 LP·zero-half 컷·병렬 LP 기반 branch-and-cut(RINS·노드별 고정·큰 문제 트리 반환·휴리스틱 정체/좁은 gap 시 휴리스틱 스레드 동적 합류 포함).
- 그 밖의 모델(분할 = 1 행, 묶음 ≤ 1 행, 다른 부가 제약): 분할 presolve + 루트 컷(zero-half·odd-hole·clique) + 병렬 LP 기반 branch-and-cut + 병렬 LNS/RINS(gap 구간별 동적 워커 배분).

## 클린룸 규칙 (요약)

다른 솔버(SCIP, CBC/CLP, HiGHS, CP-SAT, GLPK, lp_solve 등)의 **소스 코드는 열어 보지 않는다.**
Gurobi와 HiGHS는 결과 비교용 블랙박스로만 쓴다. 전체 규칙은 docs/design.md 2장.

## 빌드

```bash
python -m pip install pybind11 cmake scikit-build-core ninja   # 빌드 도구
python -m pip install -e . --no-build-isolation                 # 개발용 설치
python -I -m pytest -q tests
```

- 지금은 Mac(arm64, Apple clang)에서만 빌드·확인한다. Linux x86-64(g++) 빌드는 P0 시점 코드까지만 확인했다.
  Linux에서 확인하려면 `scripts/linux_build.sh`(docker 데몬 필요, 이 Mac에서는 `colima start` 후 실행) 또는
  Linux 머신에서 바로 `scripts/linux_native_build.sh`(docker 불필요, 작성만 하고 실행해 보지는 않음).
- C++는 표준 헤더와 `std::thread`만 쓴다(`bits/stdc++.h`, OpenMP 금지).

## 폴더

| 경로 | 내용 |
|---|---|
| `include/scsp`, `src/core` | C++ 코어: `problem`(모델), `presolve`, `sc_solver`(Lagrangian 엔진·local search·탐색·트리 반환), `lu`(기저 LU 분해), `lp`(dual·primal simplex, sifting, 이어 풀기·행 추가용 `LpSolver`), `cuts`(zero-half·odd-hole·clique 컷), `bnb`(LP 기반 branch-and-cut·RINS), `bin_solver`(일반 0-1 경로·분할 presolve·병렬 LNS/RINS) |
| `src/bindings` | pybind11 모듈 `scsp._core` (`solve_sc`, `solve_bin`, `presolve_sc`, `solve_lp`, `LpSolver`) |
| `python/scsp/lp.py` | `scsp.solve_lp(model, method)` — LP 완화, 행 쌍대값과 reduced cost. `scsp.LpSolver` — 이어 푸는 LP |
| `python/scsp/solve.py` | `scsp.solve(model, time_limit, threads, seed, warm_start, ...)` |
| `python/scsp/dms.py` | DMS drop-in 어댑터 `solveSP` (`HighsSP.solveSP`와 같은 인자·반환값) |
| `python/scsp/model.py` | 공통 모델 표현 `Model` (집합 행 CSR + 부가 제약 CSR) |
| `python/scsp/io` | 리더: OR-Library SCP(행 방식·rail 열 방식), OR-Library SPP, MPS |
| `python/scsp/gen` | 합성 생성기: DMS형 SC, 작은 SP, 표준 스위트 |
| `python/scsp/bench` | 러너(Gurobi, HiGHS, scsp), 실행기, 지표, 크기 표, BKS 표, 두 설정의 승/무/패 비교(`compare`), LP 엔진과 HiGHS LP 대조(`lpcheck`) |
| `benchmarks/` | `instance_sizes.csv`, `bks.csv`, `bks_literature.csv`, `synthetic_manifest.csv`, `results/` |
| `benchmarks/instances/` | 내려받거나 생성한 인스턴스 (git 제외) |
| `scripts/` | 다운로드, Linux 빌드(docker 방식과 직접 빌드 방식), 기준 실행 스크립트 |

## 벤치마크 하네스

```bash
scripts/download_instances.sh            # OR-Library SCP·SPP, MIPLIB air04/air05/nw04 (44 MB)
python -I -m scsp.gen.suite              # 합성 인스턴스 25개 생성 (--check 로 재현 확인)
python -I -m scsp.bench.sizes            # benchmarks/instance_sizes.csv
scripts/run_baselines.sh                 # Gurobi·HiGHS 기준 실행 (4스레드, seed 3개, 60초)
python -I -m scsp.bench.bks              # benchmarks/bks.csv (문헌값 + 실행 결과 병합)
python -I -m scsp.bench.metrics          # benchmarks/results/summary_*.csv, baseline_report.md
```

우리 솔버를 같은 하네스로 재고 기준 솔버와 비교하려면:

```bash
python -I -m scsp.bench.run --configs scsp --sets dms scp4 scp5 scp6 scpa scpb scpc scpd scpe scpnre scpnrf scpnrg scpnrh rail --time-limits 30 --seeds 1 2 3
python -I -m scsp.bench.run --configs scsp --sets sppnw sppaa sppus sppkl sp --time-limits 30 --seeds 1 2 3
python -I -m scsp.bench.run --configs scsp-p1 --sets dms --time-limits 30 --seeds 1 2 3     # P1 휴리스틱만(bnb=False)
python -I -m scsp.bench.compare --ours scsp --ref highs --out benchmarks/results/p1_vs_highs
python -I -m scsp.bench.compare --ours scsp --ref gurobi --out benchmarks/results/p1_vs_gurobi
```

LP 엔진을 모든 인스턴스에서 HiGHS LP와 대조하려면(P2 게이트):

```bash
python -I -m scsp.bench.lpcheck          # benchmarks/results/lp_check.{jsonl,csv,md}
```

일부만 실행하려면:

```bash
python -I -m scsp.bench.run --configs highs --sets scp4 scpe --time-limits 60 --seeds 1 2 3
python -I -m scsp.bench.run --configs gurobi highs --instances sp_m200_n2000_s1 --time-limits 5 10 30 60
```

- 실행 결과는 `benchmarks/results/runs/<config>.jsonl`에 한 줄에 한 실행씩 쌓인다. 이미 있는 조합은 건너뛴다.
- 각 실행은 별도 프로세스에서 돌고, 솔버가 돌려준 해는 `Model`로 다시 검사한다(`verified`).
- trace 형식: `[초, 목적값, 하한, 종류]`. 종류는 `sol`(개선 해 콜백), `bound`(하한 표본), `end`(종료 시점).
- Gurobi는 크기 제한 라이선스라 변수 ≤ 2000 그리고 제약 ≤ 2000인 모델만 실행하고 나머지는 건너뛴다.
- 내려받은 파일은 신뢰하지 않는 데이터로 다룬다. 읽는 Python은 `-I`(격리 모드)로 실행한다.

## Python에서 모델 다루기

```python
from scsp import Model
from scsp.io import read_scp, read_spp, read_mps_model
from scsp.gen import generate_dms, generate_sp

m = read_scp("benchmarks/instances/orlib_scp/scp41.txt")
m = generate_dms(n_orders=100, n_bundles=2000, seed=1)      # K개 커버 행 + 라이더 타입 ≤ 제약 3개
m.n_rows, m.n_cols, m.n_side, m.kind                        # (100, 2000, 3, "SC+side")
A, lower, upper = m.to_scipy()                              # 집합 행 다음에 부가 제약 행
m.is_feasible(x), m.objective(x)
```

## 풀기

```python
import scsp
res = scsp.solve(m, time_limit=10.0, threads=4, seed=1, warm_start=x0)   # x0: 0/1 벡터 또는 열 인덱스 목록(생략 가능)
res.status          # "optimal"(최적 증명) | "time_limit" | "no_solution" | "infeasible"
res.x, res.cols, res.obj, res.bound, res.gap
res.trace           # [(초, 목적값, 하한, "sol" | "bound"), ...]
```

- `res.stats["path"]`는 `"sc"`(커버 모델) 또는 `"bin"`(그 밖). `res.stats`에 루트 LP 값(`lp_root`), 트리 노드 수(`bnb_nodes`), 트리 상태(`bnb_status`)가 들어 있다.
- `bnb=False`를 주면 커버 모델을 P1 휴리스틱만으로 푼다(스레드 4개 모두 휴리스틱).
- 실행 가능한 warm start를 주면 반환해는 그보다 나쁘지 않다.
- 4스레드는 seed가 다른 탐색 4개가 최고해를 공유하는 방식이라, 같은 seed라도 실행마다 결과가 조금 다를 수 있다.

LP 완화만 풀려면:

```python
lp = scsp.solve_lp(m, method="auto")      # "auto" | "dual" | "sifting"
lp.status, lp.obj                          # "optimal" | "infeasible" | "unbounded" | "time_limit" | ...
lp.x, lp.row_dual, lp.red_cost             # red_cost = cost - A' row_dual, 행 순서는 m.to_scipy()와 같음
lp.stats                                   # 반복 수, 재분해 횟수, sifting 라운드, 작업 열 수
```

LP를 올려 둔 채 경계를 바꿔 가며 이어 풀려면(B&B가 쓰는 인터페이스):

```python
lp = scsp.LpSolver(m)                      # LP 완화. 일반 LP는 scsp.LpSolver.from_arrays(...)
lp.solve()                                 # "optimal" | "infeasible" | "unbounded" | "limit" | "failed" | "cutoff"
lp.obj, lp.x, lp.red_cost, lp.row_dual()
lp.set_col_bounds(j, 1.0, 1.0)             # 열 고정. 다음 solve()는 직전 기저에서 이어 푼다
basis = lp.get_basis(); lp.set_basis(basis)
lp.set_cutoff(z)                           # dual 목적값이 z를 넘으면 "cutoff". None이면 해제
```

컷은 `lp.add_rows(row_ptr, col_idx, val, row_lo, row_up)`로 넣는다(행 단위 CSR, 열 번호는 올린 열 기준). 다음 `solve()`가
직전 기저에서 dual simplex로 이어 푼다. `lp.remove_rows([...])`는 지울 행이 모두 느슨하면(`lp.row_status()`가 0) 기저를 유지한다.

DMS 코드에서는 import 한 줄만 바꾼다(`myalgorithm.py`는 이 저장소에서 건드리지 않았다).

```python
from scsp.dms import solveSP      # 기존: from HighsSP import solveSP
best_sol, sol_edge, sol_rider, init_bundle, obj = solveSP(
    initial_sol, constraint_helper, riders_available, bundle_vectors, timelimit, verbose, dist_mat, riders, heur_time)
```
