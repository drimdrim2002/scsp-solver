# SC/SP 전용 솔버 설계 계획 (초안 v0)

작성일: 2026-11-17(상태 갱신 2026-11-19) · 상태: **P0(하네스), P1(SC 휴리스틱), P2(LP 엔진), P3(정확해법), P4(B&C 강화) 구현 완료 — 결과는 `docs/p0_report.md` ~ `docs/p4_report.md`**, P5는 미착수(사용자 승인 후 시작)

---

## 1. 목표와 확정된 범위

| 항목 | 결정 |
|---|---|
| 솔버 목표 | **정확해법 + 휴리스틱 둘 다**: branch-and-cut으로 최적성을 증명하고, 제한 시간 안의 해 품질도 강하게 |
| 문제 형태 | **SC(≥1) / SP(=1) / packing(≤1) 행 + 소수의 일반 선형 부가 제약**(예: DMS 라이더 타입별 대수 ≤ 가용 대수), 변수는 모두 0/1 |
| 언어·LP | **C++17, LP(dual simplex)까지 직접 구현**, pybind11로 Python 바인딩 |
| 하드웨어 | **4스레드**, Mac(arm64, Apple clang)과 Linux x86(g++) 모두 빌드 |
| 위치 | 새 폴더 `~/optichallenge/scsp-solver` + git. DMS 폴더는 읽기만 함 |
| DMS 연동 | 범용 Python API + `HighsSP.solveSP`와 입출력이 같은 **drop-in 어댑터**. `myalgorithm.py` 연결은 요청 시에만 |
| 벤치마크 | 합성 DMS형, OR-Library SCP, 항공 승무원 SPP |
| 지표 | 고정 시간 목적값, 최적 증명 시간, primal integral, 최종 gap |

### 현실적인 목표 수준 (미리 밝혀 둠)
- **SC의 고정 시간 해 품질에서 highspy를 확실히 이기고 Gurobi에 근접**: 현실적인 목표. SC 전용 Lagrangian 휴리스틱(CFT 계열)은 문헌상 범용 MIP 솔버보다 빠르게 좋은 해를 내는 것으로 알려져 있음.
- **SP에서 좋은 해를 찾는 것**: 가능하지만 LP 기반 경로가 필수.
- **어려운 SP 인스턴스에서 Gurobi의 최적성 증명 시간에 견주는 것**: 상용 솔버 수십 인년 규모의 엔지니어링 영역이라 연구 수준 과제. 목표로 두되 보장하지 않음.

---

## 2. 클린룸 규칙

- **허용**: 논문, 교과서, 학위논문, Gurobi 공개 문서(파라미터 레퍼런스, 공개 발표자료), Gurobi 개발자들이 낸 공개 논문.
- **금지**: 다른 솔버의 **소스 코드** 열람 — SCIP, CBC/CLP, HiGHS, CP-SAT, GLPK, lp_solve 등. 공개돼 있어도 보지 않음.
- **외부 솔버의 허용 용도**: 결과 비교와 정답 확인용 **블랙박스 오라클**로만 사용(예: 우리 dual simplex 결과를 HiGHS LP 목적값과 대조).
- **"Gurobi 방식"을 구체화하는 방법**: Gurobi 공개 파라미터 레퍼런스를 체크리스트로 사용. 거기에 이름이 나오는 기법(Presolve 수준, CliqueCuts·ZeroHalfCuts·GomoryPasses, RINS, NoRelHeurTime, Sifting, MIPFocus, Concurrent 등)을 하나씩 공개 문헌의 알고리즘에 대응시킴. 특히 Gurobi 개발진의 presolve 논문(Achterberg et al. 2020)이 핵심 참고자료.
- **기록**: 구현한 각 기법마다 출처 문헌을 코드 주석과 `docs/references.md`에 남김.

---

## 3. 문제 정의

```
min  cᵀx
s.t. Σ_{j∈S_i} x_j ≥ 1      i ∈ R_cover
     Σ_{j∈S_i} x_j = 1      i ∈ R_part
     Σ_{j∈S_i} x_j ≤ 1      i ∈ R_pack
     A_side x  (≤ | = | ≥)  b_side        (소수의 일반 선형 제약, 정수/실수 계수)
     x ∈ {0,1}ⁿ
```

- DMS 모델 = `R_cover`(주문) + 3개의 `≤` 부가 제약(라이더 타입).
- 입력: Python에서 CSR/CSC numpy 배열 또는 행별 인덱스 리스트. 벤치마크용으로 OR-Library 형식 리더와 MPS 리더를 둠.
- warm start: 초기해 벡터(예: ALNS 해) 입력 지원. **반환해는 warm start보다 절대 나쁘지 않게** 보장.

---

## 4. 아키텍처

```
                    ┌──────────── Python (pybind11) ────────────┐
                    │ scsp.solve(model, time_limit, threads, …)  │
                    │ scsp.dms.solveSP(...)  ← drop-in 어댑터     │
                    └───────────────────┬────────────────────────┘
                                        │
   ┌─────────────┐   ┌──────────────────▼───────────────────┐
   │ io/          │   │ Solver 오케스트레이터 (4 std::thread)  │
   │ OR-Lib, MPS  │   │ 공유 incumbent pool · 공유 하한 · 시계   │
   └─────────────┘   └──┬──────────┬───────────┬──────────┬──┘
                        │          │           │          │
                 ┌──────▼───┐ ┌────▼─────┐ ┌───▼──────┐ ┌─▼──────────┐
                 │ presolve │ │Lagrangian│ │ LP 엔진   │ │ B&B / B&C  │
                 │+postsolve│ │ 엔진(CFT)│ │dual simplex│ │ 노드·분기·컷│
                 └──────────┘ └────┬─────┘ │ + sifting │ └─┬──────────┘
                                   │       └───┬──────┘   │
                              ┌────▼───────────▼──────────▼──┐
                              │ primal 휴리스틱 모음            │
                              │ greedy · local search · diving │
                              │ RINS · local branching · FP    │
                              └────────────────────────────────┘
```

### 4.1 두 갈래 풀이 경로 (핵심 설계 결정)
- **SC 경로 (≥1 행 + 부가 제약)**: LP 없이 바로 돌아가는 **Lagrangian 엔진**이 중심. subgradient로 Lagrangian 하한을 얻고, Lagrangian 비용 기반 greedy와 column fixing으로 해를 만듦. 시작하자마자 해와 하한(→ gap)이 모두 나옴. Gurobi의 NoRel(LP 없는 휴리스틱 단계)에 해당하는 역할.
- **SP 경로 (=1 행 포함)**: SP는 **실행 가능해를 찾는 것 자체가 NP-hard**라 greedy·반올림이 자주 실패함. 그래서 LP 기반(dual simplex → diving / feasibility pump → B&B)이 필수.
- 두 경로는 B&B 안에서 합쳐짐: SC에서도 최적성 증명은 LP 기반 B&B가 담당하고, Lagrangian 엔진은 초기해·하한·노드 휴리스틱으로 계속 쓰임.

### 4.2 모듈별 내용

**presolve / postsolve** (Achterberg et al. 2020; Balas & Ho 1980; Hoffman & Padberg 1993)
- 중복 열 제거(같은 행 집합이면 싼 열만 유지), 지배 열 제거(SC: 행 집합 ⊂ 이고 비용 ≥ 이면 제거 — 부가 제약 계수까지 비교).
- 행 지배(SC: 행 i ⊂ 행 k 이면 k 제거), singleton 행 고정.
- SP 전용: 한 =1 행이 다른 =1 행을 포함하면 차집합 열을 0으로 고정. 0으로 고정된 열의 전파.
- 모든 축소를 스택에 기록해 원래 문제의 해로 되돌림(postsolve).

**LP 엔진** (Koberstein 2005; Forrest & Goldfarb 1992; Suhl & Suhl 1990; Forrest & Tomlin 1972; Bixby et al. 1992)
- bounded dual simplex(0 ≤ x ≤ 1을 경계로 처리), primal simplex(열 추가 후 재최적화).
- 희소 LU 분해(Markowitz 피벗) + Forrest–Tomlin 업데이트, dual steepest edge 가격결정, bound-flipping ratio test.
- **sifting**: 6만 열 DMS 풀이나 수만 열 SPP처럼 열이 매우 많을 때, 작업 열 집합에서 LP를 풀고 reduced cost가 음수인 열만 추가.
- 정확성 검증: 모든 테스트 인스턴스에서 상태와 목적값을 HiGHS LP(블랙박스)와 대조.

**Lagrangian 엔진** (Beasley 1990; Caprara, Fischetti & Toth 1999)
- 커버 행을 완화하는 subgradient 최적화, 스텝 크기 조절.
- CFT 3단계: core 문제 축소 → subgradient + Lagrangian 비용 greedy → column fixing 반복 → refinement.
- 부가 제약(≤)도 같이 완화하고 승수를 관리. greedy가 부가 제약을 지키도록 처리.

**primal 휴리스틱**
- 구성: Lagrangian greedy, LP 반올림, LP diving(fractional / coefficient / guided).
- 개선: local search(SC: 1-flip·swap, 3-flip 이웃 — Yagiura, Kishida & Ibaraki 2006), RINS(Danna, Rothberg & Le Pape 2005), local branching(Fischetti & Lodi 2003).
- SP 실행 가능해: feasibility pump(Fischetti, Glover & Lodi 2005).

**절단평면** (Atamtürk, Nemhauser & Savelsbergh 2000; Padberg 1973; Caprara & Fischetti 1996)
- conflict graph: 같은 =1/≤1 행에 함께 있는 두 열은 동시에 1이 될 수 없음.
- clique cut(최대 clique 확장), odd-hole cut, zero-half({0,½}-CG) cut.
- cut pool 관리(활성도 기반 추가·제거).

**branch-and-bound**
- 노드 선택: best-bound와 depth-first를 섞은 방식.
- 변수 분기: reliability branching(Achterberg, Koch & Martin 2005).
- SP 전용: **Ryan–Foster 분기**(두 행 r, s를 같은 열이 덮는지 여부로 분기; Ryan & Foster 1981).
- reduced cost fixing, 노드 LP warm start(부모 basis 재사용), 노드에서 휴리스틱 호출 빈도 조절.

**병렬 (4 std::thread)**
- 루트 단계: Lagrangian 엔진 · LP 루트 · local search를 동시에 실행(Gurobi의 concurrent에 해당).
- B&B 단계: 공유 incumbent pool을 쓰는 노드 병렬 또는 휴리스틱 스레드 + B&B 스레드 분리. 처음엔 후자로 시작.
- 결정적 모드(같은 seed면 같은 결과) 옵션 제공.

### 4.3 Python API
```python
import scsp
m = scsp.Model(n_cols, costs)
m.add_cover_rows(row_lists); m.add_partition_rows(...); m.add_side_constraint(idx, coef, "<=", rhs)
res = scsp.solve(m, time_limit=10.0, threads=4, seed=42, warm_start=x0, mip_gap=0.0, trace=True)
res.x, res.obj, res.bound, res.gap, res.status, res.trace   # trace: [(t, obj, bound), ...]

from scsp.dms import solveSP   # HighsSP.solveSP와 같은 시그니처·반환값
```

### 4.4 빌드
- C++17 표준 헤더만 사용(`bits/stdc++.h` 금지 — Apple clang에 없음). OpenMP 대신 `std::thread`(이 Mac에 libomp 없음).
- 빌드 도구: `scikit-build-core` + `cmake` + `pybind11`. 현재 `ogc2026` 환경에 cmake·pybind11이 없으므로 **pip로 설치 필요**(승인 후).
- Linux x86 확인: 이 Mac에 `docker`와 `colima`가 설치돼 있음. `--platform linux/amd64` 컨테이너로 g++ 빌드·테스트(데몬 실행 상태는 아직 확인 안 함).

---

## 5. 벤치마크 계획

### 5.1 Gurobi 비교 가능 범위 (실측)
- 크기 제한 라이선스: **변수 ≤ 2000 그리고 제약 ≤ 2000이면 실행, 어느 하나라도 2001이면 실패** (2026-11-17 `ogc2026`에서 실측).
- 따라서 기준은 "주문 50개"가 아니라 **열(번들) 수 ≤ 2000, 제약 수 ≤ 2000**. presolve 전 원래 모델 기준이고, 제약 수에는 부가 제약도 들어간다(DMS형은 K + 3 ≤ 2000).

### 5.2 인스턴스 세트
크기는 P0에서 **실제 파일을 읽어 확인한 값**이다(2026-11-17, `python -m scsp.bench.sizes` → `benchmarks/instance_sizes.csv`에 인스턴스별 전체 표). 밀도는 nnz / (행×열).

| 세트 | 개수 | 크기(행×열) | 밀도 | 최적/최고값 출처 | Gurobi 비교 |
|---|---|---|---|---|---|
| scp4x | 10 | 200×1000 | 2% | Beasley 1987 | 가능 |
| scp5x | 10 | 200×2000 | 2% | Beasley 1987 | 가능(경계값 — 2000은 통과 실측) |
| scp6x | 5 | 200×1000 | 5% | Beasley 1987 | 가능 |
| scpa / scpb | 5 / 5 | 300×3000 | 2% / 5% | Beasley 1987 | 불가 → highspy |
| scpc / scpd | 5 / 5 | 400×4000 | 2% / 5% | Beasley 1987 | 불가 → highspy |
| scpe | 5 | 50×500 | 20% | Beasley 1987 | 가능 |
| scpnre / scpnrf | 5 / 5 | 500×5000 | 10% / 20% | Beasley 1990(휴리스틱 값), 이후 문헌 최고값 | 불가 → highspy |
| scpnrg / scpnrh | 5 / 5 | 1000×10000 | 2% / 5% | 같음 | 불가 → highspy |
| rail507 / rail516 / rail582 | 3 | 507×63009 / 516×47311 / 582×55515 | 1.2~1.3% | 문헌 최고값 | 불가 → highspy (sifting 시험용) |
| sppnw01–43 | 43 | 17~145행 × 197~288,507열 | 6~39% | Hoffman & Padberg 1993 | **24개만 가능**(열 ≤ 2000: nw08, 10, 12, 15, 20–28, 32, 34–43) |
| sppaa01–06 | 6 | 426~825행 × 5,198~8,904열 | 1.0~1.7% | 같음 | 불가 → highspy |
| sppus01–04 | 4 | 77~163행 × 13,635~1,053,137열 | 7~18% | 같음 | 불가 → highspy |
| sppkl01–02 | 2 | 55×7,479 / 71×36,699 | 14% / 8% | 같음 | 불가 → highspy |
| MIPLIB air04 / air05 / nw04 | 3 | 823×8904 / 426×7195 / 36×87482 | 1.0% / 1.7% / 20% | MIPLIB 최적값 | 불가 → highspy |
| 합성 DMS형 SC | 17 | K=50~2000 주문 × 500~60,000 번들, 부가 제약 3개 | 0.1~5% | Gurobi(작은 것) / 기준 실행 최고값 | 번들 ≤ 2000인 10개 |
| 합성 SP | 8 | 50~400행 × 500~10,000열 | 1~9% | Gurobi / 기준 실행 최고값 | 열 ≤ 2000인 7개 |

- 합계 156개(내려받은 것 131개 + 합성 25개). Gurobi 크기 제한 안에 드는 것은 71개.
- **MIPLIB 3개는 OR-Library SPP와 같은 문제다**: air04 = sppaa01, air05 = sppaa04, nw04 = sppnw04(행·열·nnz·비용 분포가 같음을 리더로 확인). MPS 리더 검증용으로만 쓰고, 집계에서는 `miplib` 세트를 따로 두어 이중 계산하지 않는다.
- 모든 OR-Library/MIPLIB 인스턴스의 비용은 정수, 합성 DMS는 실수(번들 비용 / K)다.
- 내려받은 것(2026-11-17, 사용자 승인 후): OR-Library SCP 73개 파일, SPP 55개 파일, MIPLIB 3개 파일, 합계 44 MB(`scripts/download_instances.sh`, `benchmarks/instances/`는 git 제외). rail2536 이상과 scpclr/scpcyc는 받지 않았다.
- 선택 사항: `ogc2026`에 이미 있는 OR-Tools CP-SAT를 크기 제한 없는 추가 비교 대상으로 사용.

### 5.3 측정 방식
- 조건: 같은 머신(Apple M4, 10코어), **모든 솔버 4스레드**, 같은 시간 제한, seed 3개.
- 시간 제한: 5 / 10 / 30 / 60초(DMS 사용 구간). **실행은 (솔버, 인스턴스, seed)마다 60초 한 번**만 하고, 5/10/30초 값은 그 실행의 trace에서 읽는다(시간 제한별로 따로 돌리면 기준 실행만 약 5시간이 걸림). 일부 인스턴스에서 시간 제한을 따로 준 실행과 trace 읽기를 비교해 검증한다(`docs/p0_report.md`). 최적 증명용 600초 실행은 P3에서 한다.
- 실행은 한 번에 하나씩, 실행마다 별도 프로세스로 한다(Apple M4는 성능 코어가 4개라 4스레드 실행 두 개가 겹치면 시간이 왜곡됨).
- 해 추적: Gurobi는 `MIPSOL` 콜백, HiGHS는 `kCallbackMipImprovingSolution` 콜백(highspy 1.16에 존재 확인), 우리 솔버는 내장 trace.
- 지표: ① 시간별 목적값(최고값 대비 %), ② 최적 증명 시간(shifted geometric mean, shift 10초)과 해결 개수, ③ primal integral(Berthold 2013), ④ 최종 gap.
- 인스턴스마다 최고값(BKS)을 `benchmarks/bks.csv`로 관리.

---

## 6. 구현 단계와 통과 기준

| 단계 | 내용 | 통과 기준 (go/no-go) |
|---|---|---|
| **P0 하네스** | 폴더·git·빌드 골격, OR-Lib/MPS 리더, 합성 생성기, Gurobi·HiGHS 기준 실행과 trace 저장, BKS 표 | 기준 솔버 결과가 재현되고, 인스턴스 크기·BKS 표 확정 |
| **P1 SC 휴리스틱** | 모델·presolve 기본, Lagrangian(CFT) 엔진, local search, 부가 제약 처리, DMS 어댑터 | 합성 DMS + scp 세트에서 **5/10/30초 목적값이 highspy보다 좋은 인스턴스 ≥ 80%**, Gurobi 비교 세트에서 평균 격차 보고 |
| **P2 LP 엔진** | dual/primal simplex, 희소 LU, sifting | 전 인스턴스에서 HiGHS LP와 목적값 상대오차 ≤ 1e-6, 속도는 HiGHS LP의 몇 배 이내인지 보고 |
| **P3 정확해법** | B&B, reduced cost fixing, reliability·Ryan–Foster 분기, diving, RINS, feasibility pump | scp4x/6x와 작은 SPP를 최적 증명, 모든 SPP에서 실행 가능해 확보 |
| **P4 B&C 강화** | conflict graph, clique·odd-hole·zero-half cut, 강화 presolve | 루트 gap 감소와 노드 수 감소를 수치로 보고 |
| **P5 병렬·튜닝** | 4스레드 concurrent, 파라미터 튜닝, 최종 비교 리포트 | 전체 지표 리포트 발행 |

P1을 먼저 하는 이유: DMS 사용 방식(SC, 짧은 시간)에 바로 쓰이고, highspy 대비 효과를 가장 빨리 측정할 수 있음.

---

## 7. 주요 위험

| 위험 | 대응 |
|---|---|
| 직접 짠 dual simplex의 수치 안정성·속도 | HiGHS LP 오라클로 매 인스턴스 대조, Koberstein 2005의 안정화 기법 적용, 문제 시 P3 이후로 튜닝 |
| SP 실행 가능해 탐색 실패 | feasibility pump + diving + Ryan–Foster 분기, 실패 시 warm start 해 반환 |
| 열이 매우 많은 인스턴스(6만~100만 열)에서 메모리·시간 | sifting, core 문제 축소(CFT) |
| Linux x86 빌드 차이 | colima/docker amd64 컨테이너로 매 단계 확인 |
| 클린룸 위반 위험 | 금지 목록 명시, 출처 기록 의무화 |

---

## 8. 참고문헌 (2026-11-17 Crossref·출판사 기록과 대조함 — 자세한 내용은 `docs/references.md`)

- Achterberg, T., Bixby, R.E., Gu, Z., Rothberg, E., Weninger, D. (2020). Presolve reductions in mixed integer programming. *INFORMS J. Computing* 32(2), 473–506. doi:10.1287/ijoc.2018.0857
- Achterberg, T., Koch, T., Martin, A. (2005). Branching rules revisited. *Operations Research Letters* 33(1), 42–54. doi:10.1016/j.orl.2004.04.002
- Atamtürk, A., Nemhauser, G.L., Savelsbergh, M.W.P. (2000). Conflict graphs in solving integer programming problems. *EJOR* 121(1), 40–55. doi:10.1016/S0377-2217(99)00015-6
- Balas, E., Ho, A. (1980). Set covering algorithms using cutting planes, heuristics, and subgradient optimization: a computational study. *Math. Programming Study* 12, 37–60. doi:10.1007/BFb0120886
- Beasley, J.E. (1987). An algorithm for set covering problem. *EJOR* 31(1), 85–93. doi:10.1016/0377-2217(87)90141-X  (출판사 기록의 제목은 단수 "problem"; 저자 목록과 OR-Library 페이지는 복수 "problems"로 적음)
- Beasley, J.E. (1990). A Lagrangian heuristic for set-covering problems. *Naval Research Logistics* 37(1), 151–164. doi:10.1002/1520-6750(199002)37:1<151::AID-NAV3220370110>3.0.CO;2-2
- Berthold, T. (2013). Measuring the impact of primal heuristics. *Operations Research Letters* 41(6), 611–614. doi:10.1016/j.orl.2013.08.007
- Bixby, R.E., Gregory, J.W., Lustig, I.J., Marsten, R.E., Shanno, D.F. (1992). Very large-scale linear programming: a case study in combining interior point and simplex methods. *Operations Research* 40(5), 885–897. doi:10.1287/opre.40.5.885
- Caprara, A., Fischetti, M. (1996). {0,½}-Chvátal–Gomory cuts. *Math. Programming* 74(3), 221–235. doi:10.1007/BF02592196
- Caprara, A., Fischetti, M., Toth, P. (1999). A heuristic method for the set covering problem. *Operations Research* 47(5), 730–743. doi:10.1287/opre.47.5.730
- Chu, P.C., Beasley, J.E. (1998). Constraint handling in genetic algorithms: the set partitioning problem. *J. Heuristics* 4(4), 323–357. doi:10.1023/A:1008668508685
- Danna, E., Rothberg, E., Le Pape, C. (2005). Exploring relaxation induced neighborhoods to improve MIP solutions. *Math. Programming* 102(1), 71–90. doi:10.1007/s10107-004-0518-7
- Fischetti, M., Glover, F., Lodi, A. (2005). The feasibility pump. *Math. Programming* 104(1), 91–104. doi:10.1007/s10107-004-0570-3
- Fischetti, M., Lodi, A. (2003). Local branching. *Math. Programming* 98(1–3), 23–47. doi:10.1007/s10107-003-0395-5
- Forrest, J.J., Goldfarb, D. (1992). Steepest-edge simplex algorithms for linear programming. *Math. Programming* 57(1–3), 341–374. doi:10.1007/BF01581089
- Forrest, J.J.H., Tomlin, J.A. (1972). Updated triangular factors of the basis to maintain sparsity in the product form simplex method. *Math. Programming* 2(1), 263–278. doi:10.1007/BF01584548
- Hoffman, K.L., Padberg, M. (1993). Solving airline crew scheduling problems by branch-and-cut. *Management Science* 39(6), 657–682. doi:10.1287/mnsc.39.6.657
- Koberstein, A. (2005). *The dual simplex method, techniques for a fast and stable implementation.* Dissertation, Universität Paderborn, Fakultät für Wirtschaftswissenschaften. (DOI 없음; 2차 근거로 확인)
- Padberg, M.W. (1973). On the facial structure of set packing polyhedra. *Math. Programming* 5(1), 199–215. doi:10.1007/BF01580121
- Ryan, D.M., Foster, B.A. (1981). An integer programming approach to scheduling. In A. Wren (ed.), *Computer Scheduling of Public Transport: Urban Passenger Vehicle and Crew Scheduling*, North-Holland, Amsterdam, 269–280. (DOI 없음; 인용한 논문 3편의 참고문헌 목록으로 확인)
- Suhl, U.H., Suhl, L.M. (1990). Computing sparse LU factorizations for large-scale linear programming bases. *ORSA J. Computing* 2(4), 325–335. doi:10.1287/ijoc.2.4.325
- Yagiura, M., Kishida, M., Ibaraki, T. (2006). A 3-flip neighborhood local search for the set covering problem. *EJOR* 172(2), 472–499. doi:10.1016/j.ejor.2004.10.018
- Gurobi Optimization. *Gurobi Optimizer Reference Manual* — Parameters 장(공개 문서). https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html

---

## 9. 시작 전에 필요한 승인
1. 폴더 이름 `~/optichallenge/scsp-solver` 확정
2. `ogc2026` 환경에 `pybind11`, `cmake`, `scikit-build-core` pip 설치
3. 벤치마크 다운로드: OR-Library SCP 전체, SPP 전체(약 124 MB), MIPLIB air04·air05·nw04
