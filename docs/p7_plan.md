# P7 성능 고도화 계획 (노드 컷 · 충돌 학습 · 트리 내 Sifting · 부가제약 WLS · 이식성)

작성일: 2026-11-23 · 출발 기준: P6 (`6f52ebd`, `docs/p6_report.md`) · 상태: **계획 수립(미착수)**

---

## 1. 배경 및 P6에서 드러난 핵심 병목

P6까지 완료한 결과, 153개 전체 벤치마크(4스레드, 30초)에서 **132/153개 최적 증명**(600초 포함 134/153)과 **HiGHS 대비 증명 시간 5.13배 단축 · 전 문제군(SC/SPP/Warm DMS) 5·10·30초 0패**를 달성했다. 그러나 Gurobi 및 600초 장기 증명 결과와 대조하면 다음 4가지 구조적 병목이 뚜렷하게 남아 있다.

| 병목 영역 | P6 실측 근거 (`docs/p6_report.md`, `scsp-600s.jsonl`) | 근본 원인 |
|---|---|---|
| **① 트리 중간 하한 상승 속도** | Gurobi 비교 67개 총 시간 **24.33초 vs 15.26초**(기하평균 비 `1.126`). `sp_m400_n2000_s1`에서 `scsp` **600초** 하한(`105,592`) ≈ Gurobi **60초** 하한(`105,705.7`). `dms_K200~500` 1.5~2.2배 격차 | 절단평면(`zero-half`, `clique`, `GMI`)이 **루트에서만** 생성되고 트리 내부 노드 컷이 없음. 탐색 중 발견한 분기 충돌 원인도 학습·재사용하지 못함 |
| **② 6만 열 초대형 LP 노드 처리량** | 600초 기준 `rail507`(63,009열) **7,698노드(~13노드/초)**, `dms_K2000_n60000` **2,872노드(~4.8노드/초)** | 루트 `global_fix` 이후에도 수만 개 비기저 열이 `LpSolver` 안에 남아 매 노드 BFRT/pricing 비용이 큼. 루트 probing·트리 내 열 비활성화 부재 |
| **③ 부가 제약 `dms`의 WLS 미적용** | 순수 SC(`scpnrg/h`, `rail507`)는 P6-4 WLS로 **전량 BKS 0.00%** 도달했으나, `ns > 0`인 대형 `dms`(`K500~2000, n=30k~60k`)에서는 WLS를 끔(`0.06~0.54%` 여지) | `wls_step`이 커버 행 위반 벌점만 갖고 `≤` 용량 제약 초과에 대한 라그랑주/동적 벌점 수리 경로가 없음 |
| **④ 장기 증명 시 휴리스틱 조기 소멸 & 이식성** | `rail507` 600초(`tree_giveup=0`)에서 gap `175 vs 173`(1.14%) 때문에 전 스레드가 B&B로 쏠려 `174`를 못 찾은 채 종료(30초 기본은 5/5 `174` 도달). Linux x86-64(`g++`)는 P0 이후 미검증 | `want_help_tree()`가 좁은 gap에서 휴리스틱 스레드를 남기지 않고 100% B&B에 흡수함. C++17 코어의 x86/CI 회귀 점검 필요 |

---

## 2. 공통 원칙 (클린룸 · 검증 · 게이트 규칙)

1. **클린룸 규칙 준수 (`docs/design.md` 2장)**:
   - 외부 솔버(SCIP, CBC/CLP, HiGHS, CP-SAT, GLPK, lp_solve 등)의 소스 코드는 일절 열람하지 않는다.
   - 모든 신규 기법은 공개 논문·교과서만 참고하고, Crossref/DOI 대조 결과를 `docs/references.md`에 즉시 기록한다.
2. **단계별 회귀 검증 의무**:
   - 매 하위 단계(`P7-1` ~ `P7-4`)마다 `-Wall -Wextra -Wpedantic -Werror` (`SCSP_WERROR=ON`) 빌드 통과.
   - 전체 `pytest -q tests --ignore=tests/test_ortools.py` 통과 + 단계별 신규 단위 테스트 `tests/test_p7.py` 추가.
   - `lp.cpp`/`lu.cpp` 수정 시 `python -I -m scsp.bench.lpcheck` 156/156 무오류(상대오차 ≤ `1e-6`) 재확인.
   - 집중 벤치마크(LP 19종 / B&B 증명 16종 / 대형 SPP 6종 / 대형 미증명 SC 10종)에서 악화가 없는지 확인한 뒤에만 다음 단계로 진행.
3. **전체 게이트 승인**:
   - 전체 621회 게이트 벤치마크(실행 전 기존 `scsp` 결과를 `scsp-p6{,@indep}.jsonl`로 보관) 및 장기 증명 재실행은 집중 비교 결과를 공유·승인받은 뒤 진행한다.

---

## 3. 단계별 구현 로드맵 (ROI 순)

### P7-1. 얕은 깊이 노드 컷 + Cut Aging (`src/core/bnb.cpp`, `src/core/cuts.cpp`, `src/core/lp.cpp`)
* **목표**: 루트 이후 분지된 부분 문제에서 하한을 빠르게 닫아 `dms_K200~500`, `sp_m100~400`, `scpnrg` 증명/하한 성능 강화.
* **참고 문헌**: Balas, Ceria, Cornuéjols & Natraj (1996, `doi:10.1016/0167-6377(96)00007-7`), Caprara & Fischetti (1996, `doi:10.1007/BF02592196`), Balas, Ceria & Cornuéjols (1996, *Management Science* — lift-and-project / cut validity in B&C).
* **세부 작업**:
  1. **노드별 로컬 유효성 보장**:
     - 전역 유효(globally valid) 컷만 남기거나(분지 고정 변수를 반영하지 않고 원본/전역 고정 기준으로 유도), 로컬 유효 컷(현재 `cur.fix` 기준으로 유도된 GMI / zero-half)은 **현재 워커가 해당 서브트리를 벗어날 때(`backtrack`/`switch_to`) `lp->remove_rows(base_rows)`로 자동 복구**하도록 스택 기반 컷 스코프 구현.
     - 직관적으로 구현·검증이 안전하고 전역 재사용도 가능한 **루트 재분리 + 얕은 깊이(`depth <= 4`, 큐에서 꺼낸 노드 중 `bound <= thr`이고 일정 노드 간격마다) 로컬 GMI(최대 5~8개)·zero-half 1라운드**부터 적용.
  2. **Cut Aging & LP 밀도 제어**:
     - 노드 LP 재최적화 직후 dual status가 비기저(slack > 0, `rs[i] == 0`)인 로컬 컷을 2회 연속 유휴 시 즉시 제거(`remove_rows`)해 태블로·LU 비대화 차단.
     - 컷 추가 후 시간/노드 대비 하한 증가가 기준 미달이면 자동 비활성화(P6-2 롤백 기법의 노드 확장판).
* **기대 효과**: 공통 67개 Gurobi 대비 증명 시간 기하평균 **1.126 → 1.00 이하** 도전, `sp_m200_n2000_s1` 등 중형 SPP 30초 내 증명 도전.

---

### P7-2. 충돌 그래프 · 학습 절(Conflict Clause Learning) & 루트 프로빙 (`src/core/bnb.cpp`, `src/core/presolve.cpp`, `src/core/bin_solver.cpp`)
* **목표**: 탐색·강분기 중 발견된 모순을 전역 공유 지식(2-clique / 짧은 절)으로 축적해 불필요한 노드 폭발 차단.
* **참고 문헌**: Savelsbergh (1994, `doi:10.1287/ijoc.6.4.445` — probing), Atamtürk, Nemhauser & Savelsbergh (2000, `doi:10.1016/S0377-2217(99)00015-6` — conflict graphs), Achterberg (2007, *Discrete Optimization* — conflict analysis in MIP).
* **세부 작업**:
  1. **루트 프로빙(Root Probing)**:
     - 루트 LP 직후 분수해(`0 < x_k < 1`) 및 reduced cost가 작은 상위 후보 열(`nk <= 8000` 또는 상위 150개)을 대상으로 `x_k = 1` / `x_k = 0` 도메인 전파만(`propagate`, LP 호출 없이 O(nnz) 비용) 시험 실행.
     - 모순 시 전역 0/1 고정(`global_fix`), 양쪽 프로브 결과 공통 고정 추출, `x_a = 1 ⇒ x_b = 0` 이진 충돌 쌍 수집.
  2. **탐색 중 충돌 절 학습 (`ConflictPool`)**:
     - `Worker::start_probe` 또는 `propagate` 실패 시, 고정 스택(`cur.fix`)의 분기 리터럴 중 모순에 기여한 **짧은 충돌 절(길이 2~6)**을 워커 간 `mutex` 보호(또는 주기적 동기화) `ConflictPool`(상한 예: 최대 4,096절, 길이 짧은 순 유지)에 등록.
     - 길이 2 절(`¬l_a ∨ ¬l_b`)은 행렬 2-cliques와 동일한 감시 목록으로 `propagate()`에 O(deg) 즉시 반영.
  3. **2-열 지배 전처리(선택적 검증)**:
     - SPP(`presolve_bin`)에서 비용·지원 행이 겹치는 대칭/중복 패턴 축약 추가(단, 전처리 시간이 루트 LP의 10%를 넘지 않도록 상한 설정).
* **기대 효과**: `sp_m100~400`, `scpnre/f`, `dms_tight` 노드 수 **20~40% 추가 감소**.

---

### P7-3. 6만 열 대형 문제용 트리 내 Sifting / Column Aging (`src/core/bnb.cpp`, `src/core/lp.cpp`)
* **목표**: `rail507`(63,009열), `dms_K500~2000`(3만~6만 열), `sp_m300_n10000`의 **노드당 LP 속도를 2~4배** 가속.
* **참고 문헌**: Bixby et al. (1992, `doi:10.1287/opre.40.5.885` — sifting), Forrest & Goldfarb (1992).
* **세부 작업**:
  1. **대형 트리용 동적 작업 열 집합 (`LpSolver` / `Search`)**:
     - `nk > 8000`인 대형 모델에서 루트 `global_fix` 이후에도 활성 열이 많으면, **기저 열 + reduced cost 상위 `max(4*m, 4000)`열**만 활성 경계(`ub = 1`, 나머지 비활성 열은 `ub = 0` 고정)로 두고 B&B 노드를 빠르게 탐색.
     - 전역 하한 또는 열린 큐 상단 노드를 처리할 때만 비활성 열들의 exact pricing(`d_j = c_j - y^T A_j`)을 검사해 dual 불만족 열(`d_j < -1e-7`)이 있으면 작업 집합에 편입 후 재최적화(정확성 100% 보존: **비활성 열 포함 reduced cost 검사 통과 없이는 가지치기·최적 선언 금지**).
  2. **강분기·노드 LP 미세 튜닝**:
     - 대용량 열에서 strong branching 후보 수(`sblim`) 및 `strong_iters` 적응 조절.
* **안전 장치**: 비활성 열 pricing 누락은 하한 오류(invalid bound)를 유발하므로, `global_bound` 갱신·가지치기·`EXHAUSTED` 판정 경로에 **전 열 reduced cost 검증 단위 테스트**(`tests/test_p7.py`)를 반드시 추가.
* **기대 효과**: `rail507` 600초 탐색 노드 `7,698 → 20,000+` 확대, `dms_K300_n20000` 30초 증명 도전, `dms_K200_n10000` 13.5초 → 5~8초대 단축.

---

### P7-4. 부가 제약(`ns > 0`) 대응 동적 행 가중치 탐색 & 장기 모드 휴리스틱 보존 (`src/core/sc_solver.cpp`)
* **목표**:
  1. 3만~6만 열 `dms` (`ns > 0`) 해 품질 추가 개선.
  2. `rail507` 같은 1-차이 대형 SC의 장기 최적 증명 달성.
* **참고 문헌**: Yagiura, Kishida & Ibaraki (2006, `doi:10.1016/j.ejor.2004.10.018`), Beasley (1990).
* **세부 작업**:
  1. **부가 제약 용량 초과 페널티 포함 WLS (`wls_step_side`)**:
     - 기존 `wls_step` 목적 함수에 커버 미달 페널티 $\sum_i rw_i \cdot \max(0, 1 - cov_i)$ 외에 부가 제약 초과 페널티 $\sum_{t=0}^{ns-1} sw_t \cdot \max(0, usage_t - rhs_t)$ 추가(`sw_t` 초기값은 `max(v_root[t], avg_col_cost / avg_side_coef)`).
     - `fits_add`의 엄격 차단을 완화해 일시적으로 `usage_t > rhs_t`인 이동도 허용하되, **공유 해(`S.offer`) 등록 및 `keep_current()` 직전에는 반드시 `side_ok()` + greedy repair(초과 제약에 걸린 비싼 열 교체)**를 통과한 100% feasible 해만 반영.
     - 대형 `dms`(`n >= 15000`)에서 기존 Worker 2 CFT 처리량을 해치지 않도록 별도 예산/주기(예: 패스 후 `0.15s` 또는 Worker 2 교대)로 실행해 회귀 방지.
  2. **장기 증명 시 `rail507` 류 1-차이 문제 휴리스틱 병행**:
     - `ns == 0`이고 `tree_giveup == 0.0`(장기 증명)이며 `P.integral`에서 `ub - ceil(lb) >= 1`인 동안에는, `want_help_tree()`가 **Worker 2(WLS 전담)까지 초기 `min(30s, 0.1 * time_limit)` 또는 정체 주기 동안은 B&B로 흡수하지 않고 유지**해 `174` 같은 최적해 발견 후 전 스레드로 증명을 닫도록 조정.
* **기대 효과**: `rail507` **600초(및 30초+α) 직접 최적 증명** 유력, 6만 열 `dms` BKS 추가 경신.

---

### P7-5. 이식성 · CI · 최종 P7 게이트 검증 (`scripts/`, `.github/workflows/`)
1. **Linux x86-64(`g++`) 빌드 및 테스트 검증**:
   - `scripts/linux_build.sh` 또는 `scripts/linux_native_build.sh`로 `-Wall -Wextra -Wpedantic -Werror` 빌드 + 전체 `pytest` 통과 확인.
2. **GitHub Actions CI 도입** (`.github/workflows/ci.yml`):
   - macOS + Linux 매트릭스에서 `SCSP_WERROR=ON` 빌드 및 단위 테스트 자동 실행.
3. **최종 게이트 벤치마크**:
   - `scsp-p6{,@indep}.jsonl` 보관 후 동일 621회 게이트 + 장기 증명 재실행, `docs/p7_report.md` 및 `README.md` 갱신.

---

## 4. P7 정량 목표 및 일정

| 단계 | 핵심 산출물 | 검증 기준 (Go/No-Go) |
|---|---|---|
| **P7-1** | `bnb.cpp`·`cuts.cpp` 노드 컷 + cut aging | 16종 증명 벤치마크 총 시간 **74.65s → 62s 이하**, Gurobi 67종 기하평균 비 **≤ 1.02**, `pytest` 전량 통과 |
| **P7-2** | `bnb.cpp`·`presolve.cpp` 루트 프로빙 + 충돌 절 학습 | 노드 수 추가 감소(특히 `sp_m100`, `scpnre/f`, `dms_tight`), 16종 + 대형 SPP 6종 악화 없음 |
| **P7-3** | `bnb.cpp`·`lp.cpp` 트리 내 sifting + 열 에이징 | 1만~6만 열 노드/초 **2배 이상**, invalid bound/pruning **0건**, `lpcheck` 156/156 통과 |
| **P7-4** | `sc_solver.cpp` 부가제약 WLS + 장기 휴리스틱 보존 | 대형 `dms` 10s/30s gap 추가 개선(악화 없음), **`rail507` 600초(또는 단축 시간) 내 최적 증명** |
| **P7-5** | CI·Linux 검증 + 621회 게이트 + `docs/p7_report.md` | **30초 증명 134+/153**, **Gurobi 67종 기하평균 비 ≤ 1.00**, HiGHS 전 구간 0패 유지, 621/621 `verified` |
