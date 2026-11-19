# scsp-solver

**Set Covering (SC) / Set Partitioning (SPP) 전용 MIP 솔버** (C++17 코어 + pybind11).  
외부 LP/MIP 라이브러리 없이 **dual/primal simplex · 희소 LU · sifting · 절단평면 · 병렬 Branch-and-Cut · Lagrangian(CFT) 휴리스틱**까지 공개 문헌만 참고해 클린룸 방식으로 직접 구현한 프로젝트입니다.

* **설계 문서**: [docs/design.md](docs/design.md)
* **단계별 리포트**: [P0 하네스](docs/p0_report.md) · [P1 SC 휴리스틱](docs/p1_report.md) · [P2 LP 엔진](docs/p2_report.md) · [P3 정확해법](docs/p3_report.md) · [P4 B&C 강화](docs/p4_report.md) · [P5 병렬 B&C·최종 리포트](docs/p5_report.md)
* **참고문헌 및 출처 기록**: [docs/references.md](docs/references.md)

---

## 1. 프로젝트 개요

### 다루는 문제 형태
모든 결정변수가 $x \in \{0, 1\}^n$인 집합 덮기·분할·묶기 + 일반 선형 부가 제약 문제입니다.

$$\min \; c^\top x + \text{offset}$$
$$\text{s.t.}\quad \sum_{j \in S_i} x_j \; (\ge \mid = \mid \le)\; 1 \quad \forall i, \qquad A_{\text{side}} x \; (\le \mid = \mid \ge)\; b_{\text{side}}, \qquad x \in \{0,1\}^n$$

### 두 갈래 풀이 아키텍처 (P0~P5 완결)
`scsp.solve()`는 모델 구조에 따라 최적의 C++17 병렬 엔진 경로를 자동 선택합니다.

| 경로 (`stats["path"]`) | 대상 모델 | 핵심 알고리즘 |
|---|---|---|
| **`sc`** | 덮기 행(`≥ 1`) + 비음수 계수 `≤` 부가 제약 (OR-Library SCP, Rail, DMS형 모델) | SC 전처리 + **3스레드 Lagrangian(CFT) 휴리스틱·로컬 탐색** + **루트 LP(sifting)·zero-half 컷·병렬 Branch-and-Cut(RINS·노드별 고정)** + 휴리스틱 정체/좁은 gap 시 휴리스틱 스레드 동적 B&B 합류, 대형 미증명 문제 시 트리 조기 반환 |
| **`bin`** | 분할 행(`= 1`), 묶기 행(`≤ 1`), 일반 부가 제약·음수 비용 포함 (OR-Library SPP, 합성 SP 등) | 분할 전처리(Borndörfer) + **루트 컷(zero-half · odd-hole · clique)** + **병렬 Branch-and-Cut** + **gap 기반 동적 병렬 LNS/RINS** (넓은 gap에서는 3 LNS, 좁은 gap에서는 최대 4스레드 병렬 B&B) |

### 클린룸 규칙
* SCIP, CBC/CLP, HiGHS, CP-SAT, GLPK, lp_solve 등 다른 솔버의 **소스 코드는 일절 열람하지 않았습니다.**
* HiGHS와 Gurobi는 결과 비교 및 LP 정확성 검증용 **블랙박스 오라클**로만 사용했습니다.
* 구현한 모든 기법의 출처 문헌과 DOI는 [docs/references.md](docs/references.md)에 기록했습니다.

---

## 2. 벤치마크 요약 (P5 최종 게이트)

### 벤치마크 구성 및 조건
* **환경**: Apple M4, **모든 솔버 4스레드**, 제한 시간 **30초** (기준 솔버 HiGHS·Gurobi는 60초 실행 후 trace에서 5·10·30초 값 추출), 프로세스당 1회 순차 실행.
* **인스턴스 (총 153개 + DMS 시작해 17개)**:
  * **SC 90개**: OR-Library `scp4`~`scpe` (50), `scpnre`~`scpnrh` (20), `rail507/516/582` (3), 합성 DMS형 `K=50~2000 × n=500~60000` (17).
  * **SPP 63개**: OR-Library 항공 승무원 스케줄링 `sppnw` (43), `sppaa` (6), `sppus` (4), `sppkl` (2) + 합성 `sp` (8).
  * **총 621회 실행**(기본 3 seed, 변동 큰 세트 5 seed, DMS warm start `near`/`indep` 각 3 seed) 전량 독립 검증 통과(**미검증 0건 · 하한 초과 0건 · 최적값 불일치 0건**).
* **Gurobi 비교 범위**: 크기 제한 라이선스 조건(원래 모델 기준 변수 ≤ 2,000 및 제약 ≤ 2,000)에 해당하는 **71개**(SC 40, SPP 31).

### 핵심 결과 (P5 vs HiGHS 1.16.0 vs Gurobi 13.0.2)

#### ① 30초 전 seed 최적 증명 수 및 속도
| 구분 | 전체 | **scsp (P5)** | scsp (P4) | scsp (P3) | HiGHS | Gurobi |
|---|---:|---:|---:|---:|---:|---:|
| **SC 전체** | 90 | **74** | 69 | 63 | 68 | 40 / 40 |
| **SPP 전체** | 63 | **58** | 56 | 54 | 52 | 27 / 31 |
| **합계** | **153** | **132** | 125 | 117 | **120** | 67 / 71 |

* **vs HiGHS**:
  * 증명 수 **132 vs 120 (+12개 우위)** — HiGHS가 30초 내 증명한 인스턴스를 **1개도 빠짐없이** scsp도 증명했습니다(`scpnre` 5/5, `scpnrf` 5/5, `sppnw/aa/us/kl` 55/55 전량 증명).
  * 양쪽 모두 증명한 120개 기준 증명 시간 합 **98.86초 vs 367.74초 (3.72× 빠름)**, 시간 비 기하평균 **0.22**, **100/120개(83.3%)에서 우위**.
* **vs Gurobi (71개)**:
  * 증명 수 **67/71 vs 67/71 완전 동률**. 공통 증명 67개 중 **35개(52.2%)에서 scsp가 더 빠르며**, 시간 비 기하평균 **1.27**.

#### ② 고정 시간 해 품질 (승 / 무 / 패, 평균 목적값 차이 %)
| 비교 대상 | 5초 | 10초 | 30초 | 비고 |
|---|---|---|---|---|
| **SC vs HiGHS (90개)** | **14 / 73 / 3 (−1.48%)** | **11 / 76 / 3 (−1.03%)** | **7 / 77 / 6 (−0.27%)** | `dms` 17개 전 구간 무패 (−2.32~−1.42%) |
| **SPP vs HiGHS (62개)** | **18 / 43 / 1 (−0.46%)** | **15 / 46 / 1 (−0.74%)** | **10 / 52 / 0 (−1.93%)** | **30초 62개 전량 무패** |
| **SC vs Gurobi (40개)** | **0 / 40 / 0 (+0.00%)** | **0 / 40 / 0 (+0.00%)** | **0 / 40 / 0 (+0.00%)** | **5·10·30초 전 구간 전량 동률** |
| **SPP vs Gurobi (31개)** | 2 / 26 / 3 (+0.22%) | 0 / 26 / 5 (+0.26%) | 0 / 27 / 4 (+0.14%) | `sppnw` 24/24 동률, 합성 `sp` +0.64% |
| **DMS 시작해 `near` vs `highs-nopre`** | **10 / 7 / 0 (−0.90%)** | **8 / 9 / 0 (−0.75%)** | **6 / 11 / 0 (−0.62%)** | Gurobi 10개 대비 10·30초 전량 동률 |
| **DMS 시작해 `indep` vs `highs-nopre`** | **10 / 7 / 0 (−1.80%)** | **8 / 9 / 0 (−1.64%)** | **7 / 10 / 0 (−1.33%)** | Gurobi 10개 대비 10·30초 전량 동률 |

#### ③ 단계별 진화 (P0 → P5)
| 단계 | 핵심 구현 | 30초 최적 증명 | 비고 |
|---|---|---:|---|
| P0 | 벤치마크 하네스·리더·합성 생성기 | – | 153개 인스턴스 · BKS · 4대 지표 체계 |
| P1 | SC Lagrangian(CFT) 휴리스틱·DMS 어댑터 | 49 / 90 (SC) | DMS·SCP 고정 시간 해 품질 HiGHS 추월 |
| P2 | 자체 C++17 dual/primal simplex · LU · sifting | – | 전 인스턴스 HiGHS LP와 상대오차 ≤ 1e-6 |
| P3 | LP 기반 B&B, reduced cost fixing, SPP 경로 | 117 / 153 | SC/SPP 통합 정확해법 구축 |
| P4 | Zero-half·clique 컷, SPP presolve, RINS/LNS | 125 / 153 | 루트 gap 50~75%·노드 70.5% 감소, HiGHS 추월 |
| **P5** | **4스레드 병렬 B&C 트리, SC/SPP 동적 워커 합류** | **132 / 153** | **HiGHS +12개 우위(3.72×), Gurobi 67/71 동률** |

자세한 세트별 표, 인스턴스별 증명 시간, Primal Integral 및 최종 gap은 [docs/p5_report.md](docs/p5_report.md)를 참고하세요.

---

## 3. 설치 및 빌드

C++17 표준 헤더와 `std::thread`만 사용합니다(`bits/stdc++.h`, OpenMP 미사용). Python 3.9+ 및 `numpy`가 필요합니다.

```bash
# 1) 빌드 도구 설치
python -m pip install pybind11 cmake scikit-build-core ninja

# 2) 개발용(editable) 빌드 및 설치
python -m pip install -e . --no-build-isolation

# 3) 단위 테스트 실행 (170개, OR-Tools 선택 테스트 제외)
python -I -m pytest -q tests --ignore=tests/test_ortools.py
```

* 경고 없는 엄격 빌드를 확인하려면 `-Ccmake.define.SCSP_WERROR=ON` 옵션을 붙입니다.
* 주 개발·검증 환경은 macOS(arm64, Apple clang)입니다. Linux x86-64(g++) 빌드는 `scripts/linux_build.sh`(Docker/Colima 필요) 또는 Linux 머신에서 `scripts/linux_native_build.sh`를 사용할 수 있습니다.

---

## 4. 사용 방법

### ① Python에서 모델 직접 만들기 및 풀기

`Model.from_rows(name, cost, rows, row_sense, side=...)`로 0-based 행별 열 목록에서 바로 모델을 구성할 수 있습니다. `row_sense`는 `"G"`(≥ 1, 덮기), `"E"`(= 1, 분할), `"L"`(≤ 1, 묶기)을 지원합니다.

```python
import scsp
from scsp import Model

# 예: 변수 4개, 덮기 행 3개 + 일반 부가 제약 1개(x0 + x1 <= 1)
m = Model.from_rows(
    name="example",
    cost=[3.0, 2.0, 4.0, 5.0],
    rows=[[0, 1], [1, 2], [2, 3]],
    row_sense="G",                              # 모든 집합 행 >= 1 (행별 리스트 ["G", "E", "L"]도 가능)
    side=[([0, 1], [1.0, 1.0], "L", 1.0)],      # (열 인덱스, 계수, "L"|"E"|"G", 우변)
)

res = scsp.solve(
    m,
    time_limit=10.0,
    threads=4,
    seed=1,
    warm_start=None,      # 선택: 0/1 벡터 또는 선택된 열 인덱스 목록 (실행 가능하면 반환해는 이보다 나쁘지 않음)
    mip_gap=0.0,          # 상대 gap 종료 기준
)

print(res.status)         # "optimal" | "time_limit" | "no_solution" | "infeasible"
print(res.obj, res.bound, res.gap)
print(res.cols, res.x)    # 선택된 열 인덱스, 0/1 해 벡터
print(res.stats["path"], res.stats["bnb_nodes"], res.stats["cuts"])
```

주요 선택 인자 (`scsp.solve`):
* `bnb=False`: SC 모델에서 LP·B&C를 끄고 4스레드 Lagrangian 휴리스틱(P1)만 실행.
* `cuts=0`: 루트 절단평면 비활성화(기본 `None`은 자동: SC `1`, 일반 0-1 경로 `7`).
* `tree_giveup`: SC에서 가망 없는 트리 스레드를 휴리스틱으로 되돌리는 배수 기준(기본 `3.0`, `0`이면 포기 안 함).
* `verify=True`: 반환해를 `Model`로 독립 재검증(`res.verified`).

### ② 파일(OR-Library / MPS) 및 합성 모델 불러오기

```python
from scsp.io import read_scp, read_spp, read_mps_model
from scsp.gen import generate_dms, generate_sp

m1 = read_scp("benchmarks/instances/orlib_scp/scp41.txt")
m2 = read_spp("benchmarks/instances/orlib_spp/sppnw04.txt")
m3 = read_mps_model("benchmarks/instances/miplib/air04.mps.gz")
m4 = generate_dms(n_orders=200, n_bundles=2000, seed=1)     # K개 덮기 행 + 라이더 타입 <= 제약 3개
m5 = generate_sp(n_rows=100, n_cols=1000, seed=1)           # 합성 Set Partitioning

A, lower, upper = m4.to_scipy()                             # SciPy CSR 행렬 변환
```

### ③ LP 완화 단독 풀이 및 증분 LP (`LpSolver`)

```python
import scsp

# 1) LP 완화 한 번에 풀기
lp = scsp.solve_lp(m4, method="auto")       # "auto" | "dual" | "sifting"
print(lp.status, lp.obj)                    # "optimal" | "infeasible" | "unbounded" | ...
print(lp.x, lp.row_dual, lp.red_cost)       # red_cost = cost - A' row_dual

# 2) 증분 LP: 열 경계 변경·컷 행 추가/삭제 후 직전 기저에서 dual simplex 재최적화
solver = scsp.LpSolver(m4)                  # 일반 LP는 scsp.LpSolver.from_arrays(...)
solver.solve()
solver.set_col_bounds(0, 1.0, 1.0)          # x_0 = 1 고정
solver.set_cutoff(lp.obj + 10.0)            # dual 목적값이 cutoff를 넘으면 즉시 "cutoff" 반환
basis = solver.get_basis()
solver.set_basis(basis)
solver.add_rows(row_ptr, col_idx, val, row_lo, row_up)
solver.remove_rows([solver.rows() - 1])
```

### ④ DMS Drop-in 어댑터

기존 `HighsSP.solveSP`와 동일한 인자·반환값을 가지므로 import 한 줄만 교체해 바로 사용할 수 있습니다.

```python
from scsp.dms import solveSP    # 기존: from HighsSP import solveSP

best_sol, sol_edge, sol_rider, init_bundle, obj = solveSP(
    initial_sol, constraint_helper, riders_available, bundle_vectors,
    timelimit, verbose, dist_mat, riders, heur_time,
)
```

### ⑤ 벤치마크 하네스 재현

```bash
# 1) 공개 인스턴스 다운로드 (44 MB) 및 합성 인스턴스 25개 생성
scripts/download_instances.sh
python -I -m scsp.gen.suite

# 2) scsp 실행 (결과는 benchmarks/results/runs/scsp.jsonl에 누적, 이미 끝난 조합은 자동 건너뜀)
python -I -m scsp.bench.run --configs scsp --sets scp4 scp5 dms --time-limits 30 --seeds 1 2 3 --threads 4
python -I -m scsp.bench.run --configs scsp --sets sppnw sppaa sp --time-limits 30 --seeds 1 2 3 --threads 4

# 3) BKS·지표 재생성 및 기준 솔버와 승/무/패 비교
python -I -m scsp.bench.bks
python -I -m scsp.bench.metrics
python -I -m scsp.bench.compare --ours scsp --ref highs --times 5 10 30 --out benchmarks/results/custom_vs_highs
python -I -m scsp.bench.compare --ours scsp --ref gurobi --times 5 10 30 --out benchmarks/results/custom_vs_gurobi

# 4) LP 엔진 정확성·속도 전수 대조 (vs HiGHS LP)
python -I -m scsp.bench.lpcheck
```

---

## 5. 디렉토리 구조

| 경로 | 내용 |
|---|---|
| `include/scsp/`, `src/core/` | C++17 코어: `problem`(모델), `presolve`(SC 전처리), `sc_solver`(Lagrangian CFT·로컬 탐색·트리 반환·동적 합류), `lu`(희소 LU · Forrest–Tomlin 업데이트), `lp`(bounded dual/primal simplex · sifting · `LpSolver` · `clone`), `cuts`(zero-half · odd-hole · clique 분리), `bnb`(병렬 Branch-and-Cut · `BnbTeam` · RINS · diving · FP · cover search), `bin_solver`(일반 0-1 경로 · 분할 presolve · gap 기반 병렬 LNS/RINS) |
| `src/bindings/` | pybind11 바인딩 `scsp._core` (`solve_sc`, `solve_bin`, `presolve_sc`, `solve_lp`, `LpSolver`) |
| `python/scsp/` | Python API (`model.py`, `solve.py`, `lp.py`, `dms.py`), 리더 (`io/`), 합성 생성기 (`gen/`), 벤치마크 하네스 (`bench/`) |
| `tests/` | 단위·통합 테스트 스위트 (170개, `test_p4.py`, `test_p5.py`, `test_lp*.py`, `test_bnb.py` 등) |
| `benchmarks/` | 인스턴스 크기표(`instance_sizes.csv`), `bks.csv`, `results/`(P0~P5 실행 원자료·비교표·지표) |
| `docs/` | 설계서(`design.md`), 단계별 보고서(`p0_report.md`~`p5_report.md`), 출처 문헌(`references.md`) |
| `scripts/` | 인스턴스 다운로드, Linux 빌드(`linux_build.sh`, `linux_native_build.sh`), 기준 실행 스크립트 |
