# 참고문헌과 출처 기록

클린룸 규칙(docs/design.md 2장)에 따라, 구현한 기법마다 근거 문헌을 여기에 적는다.
다른 솔버의 소스 코드는 출처가 될 수 없다.

## 1. 설계 문서의 참고문헌 (2026-11-17 대조)

Crossref 메타데이터와 출판사 기록으로 저자·연도·제목·권(호)·쪽을 대조했다. 23개 중 20개는 그대로 맞았고,
2개는 제목을 고쳤다(Balas & Ho 1980의 부제 누락, Beasley 1987의 제목 단·복수). Gurobi 문서는 서지 대조 대상이 아니다.
Koberstein 2005와 Ryan & Foster 1981은 원문이 아니라 2차 근거(저자 본인의 인용, 다른 논문의 참고문헌 목록)로 확인했다.
Hoffman & Padberg 1993, Caprara–Fischetti–Toth 1999, Beasley 1987·1990은 유료 논문이라 본문은 읽지 못했고 서지 정보만 확인했다.

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

## 2. BKS 표에 쓴 추가 출처

- Beasley, J.E., Jørnsten, K. (1992). Enhancing an algorithm for set covering problems. *EJOR* 58(2), 293–300. doi:10.1016/0377-2217(92)90215-U — scp4x–scpe 최적값(Tables 2–3)
- Beasley, J.E. (2024). A heuristic for the non-unicost set covering problem using local branching. *Int. Trans. Oper. Res.* 31(5), 2807–2825. doi:10.1111/itor.13446 (arXiv 2207.03248) — scp4x–scpnrh 최적/최고값(Table 2), scpnrg 최적 증명(Table 5)
- Borndörfer, R. (1998). *Aspects of Set Packing, Partitioning, and Covering.* PhD thesis, TU Berlin — SPP 55개 최적값(Table 3.4)
- Umetani, S., Yagiura, M. (2007). Relaxation heuristics for the set covering problem. *J. Oper. Res. Soc. Japan* 50(4), 350–375. doi:10.15807/jorsj.50.350 — rail516·rail582 최적값(Table 12)
- Umetani, S. (2017). Exploiting variable associations to configure efficient local search algorithms in large-scale binary integer programs. *EJOR* 263(1), 72–81. doi:10.1016/j.ejor.2017.05.025 — 세트 평균값 대조용
- Yelbay, B., Birbil, Ş.İ., Bülbül, K. (2015). The set covering problem revisited. *J. Ind. Manag. Optim.* 11(2), 575–594. doi:10.3934/jimo.2015.11.575 — 대조용
- MIPLIB 2017 인스턴스 페이지(miplib.zib.de): air04, air05, nw04, rail507
- OR-Library 안내 페이지: scpinfo.html, sppinfo.html (people.brunel.ac.uk/~mastjjb/jeb/orlib)

BKS 대조에서 나온 사실:

- scpnre5의 최적값은 28이다(설계 초안을 쓸 때 기억한 값 27은 틀림).
- scpnrg1–5는 2024년 논문에서 Cplex로 최적이 증명됐다. scpnrh1–5는 최고값만 알려져 있고 증명이 없다.
- sppaa01·sppaa04·sppus01은 Chu & Beasley 1998(Hoffman & Padberg 인용)의 표에 56138·26402·10022로 실려 있지만,
  Borndörfer 1998과 MIPLIB의 값은 56137·26374·10036이다. `benchmarks/bks.csv`는 솔버가 증명한 값으로 확정한다.
- sppnw02는 Chu & Beasley 105444, Borndörfer 102903으로 서로 다르다. 역시 솔버 증명값으로 확정한다.

## 3. 구현 기법별 출처

| 구현 | 위치 | 출처 |
|---|---|---|
| primal gap, primal integral | `python/scsp/bench/metrics.py` | Berthold 2013 |
| shifted geometric mean(shift 10초) | `python/scsp/bench/metrics.py` | MIP 벤치마크 관행. 정의: exp(mean(log(t + s))) − s |
| OR-Library 파일 형식 | `python/scsp/io/orlib.py` | OR-Library scpinfo.html, sppinfo.html |
| DMS형 모델 구성 | `python/scsp/gen/dms.py` | DMS 대회 코드 `HighsSP.py`(자체 코드, 읽기 전용 참고) |
| presolve: 단일 열 행 고정, 지배 열·중복 열 제거, 행 지배, 비용 테스트(행별 최소 비용 합) | `src/core/presolve.cpp` | Balas & Ho 1980; Beasley 1987. 부가 제약 계수까지 비교하는 조건은 지배 정의를 그대로 확장한 것 |
| Lagrangian 완화와 subgradient(스텝 = λ(UB − L)/‖s‖², 하한이 정체되면 λ 절반, 0인 승수의 음수 방향 성분 제거) | `src/core/sc_solver.cpp` `subgradient` | Beasley 1990 |
| 초기 승수 u_i = min c_j/\|I_j\|, core 문제(행마다 reduced cost 최소 5개 + 임계값 이하 열), pricing 주기 T 조절 | `sc_solver.cpp` `init_multipliers`, `price` | Caprara, Fischetti & Toth 1999 |
| Lagrangian 비용 greedy(γ/μ, γ·μ 점수), 중복 열 제거, heuristic phase(승수 열마다 greedy) | `sc_solver.cpp` `greedy`, `evaluate` | Caprara, Fischetti & Toth 1999. lazy heap은 점수가 단조 증가한다는 성질을 쓴 구현 선택 |
| column fixing(reduced cost가 음수인 열 + greedy 앞부분 고정 후 잔여 문제 반복), refinement(gap 기여 δ_j가 작은 열부터 π 비율까지 고정) | `sc_solver.cpp` `column_fix`, `three_phase`, `refine_fix` | Caprara, Fischetti & Toth 1999 |
| 부가 제약(≤)을 승수 v ≥ 0으로 함께 완화, greedy·탐색에서 남은 용량 확인 | `sc_solver.cpp` 전반 | 일반 Lagrangian 완화(Beasley 1990의 틀을 ≤ 제약에 적용). 용량 확인은 자체 설계 |
| reduced-cost fixing(하한 + reduced cost가 개선 기준을 넘으면 열 제거) | `sc_solver.cpp` `apply_rc_fix` | Beasley 1987 |
| 미커버 행 분기 깊이 우선 탐색, 노드 하한 = 남은 행으로 제한한 Lagrangian 값 | `sc_solver.cpp` `Enum` | Balas & Ho 1980(행 분기), Beasley 1987(Lagrangian 하한 트리 탐색). 승수는 루트 값 고정(노드 재최적화 없음) |
| local search: 열 하나 추가 후 중복이 된 열 제거(1-추가/다중 제거) | `sc_solver.cpp` `improve` | Yagiura, Kishida & Ibaraki 2006의 flip 이웃 중 일부만 구현(3-flip, 벌점 가중은 미구현) |
| 큰 이웃 탐색: incumbent 일부를 고정하고 나머지를 위 트리 탐색으로 정확히 풀기 | `sc_solver.cpp` `lns_step` | 고정 후 재최적화는 Caprara, Fischetti & Toth 1999의 refinement와 같은 틀. 이웃 선택(무작위 행 주변)과 정확 탐색 결합은 자체 설계 |
| SC 해 → 분할 해 변환(중복 주문을 절감이 가장 작은 묶음에 남김) | `python/scsp/dms.py` | 사용자의 DMS 코드(HighsSP.py)의 규칙을 그대로 옮김 |
| 기저 LU 분해: singleton 열·행 우선, Markowitz 수 최소 + 임계 피벗(열 최대값의 0.1배 이상) | `src/core/lu.cpp` `factor` | Markowitz 1957; Suhl & Suhl 1990. 특이 기저는 피벗이 안 된 위치를 logical 열로 바꿔 복구(일반적인 관행) |
| Forrest–Tomlin 업데이트(spike 열 교체, 행 eta), 새 대각 원소 = 피벗 원소 × 옛 대각 원소로 정확도 검사 | `lu.cpp` `update` | Forrest & Tomlin 1972 |
| bounded dual simplex: dual steepest edge 가격결정과 가중치 갱신, bound flipping ratio test, 비용 섭동, cost shifting | `src/core/lp.cpp` `Simplex::dual`, `perturb`, `fix_dual` | Koberstein 2005; Forrest & Goldfarb 1992 |
| primal simplex: Devex 가격결정, 허용오차 2단계 ratio test | `lp.cpp` `primal_loop` | Harris 1973 |
| primal의 경계 섭동과, 섭동 제거 뒤 dual simplex로 정리하는 반복 | `lp.cpp` `primal`, `perturb_bounds` | Koberstein 2005의 비용 섭동·제거 절차를 primal 쪽에 대칭으로 적용한 자체 구성 |
| sifting: 작업 열 집합의 LP를 풀고 reduced cost가 음수인 열을 추가 | `lp.cpp` `solve_lp` | Bixby et al. 1992. 첫 작업 집합(행마다 비용/길이가 작은 5열), big-M 인공 열, 새 열을 상한에 놓아 dual 실행 가능성을 지킨 채 dual simplex로 다시 푸는 방식은 자체 구성 |
| LP를 올려 둔 채 경계만 바꿔 dual simplex로 이어 풀기, 기저 저장·복원, dual 목적값 cutoff | `src/core/lp.cpp` `LpSolver`, `Simplex::set_bounds` | Koberstein 2005(경계 변경 뒤에도 기저가 dual 실행 가능하다는 성질). 정체 시 비용 섭동을 늦게 거는 규칙은 자체 구성 |
| LP 기반 branch and bound: 최선 하한 노드 선택 + 내려가기(plunging), 정수 비용이면 하한 올림 | `src/core/bnb.cpp` `Search::run` | 교과서적 방법(Land & Doig 1960 — 서지 미대조) |
| reduced cost fixing: 루트 LP 값 + reduced cost가 최고해 기준을 넘는 열 제거, 최고해가 좋아질 때마다 다시 적용. 살아남은 열만 트리의 LP에 올림 | `bnb.cpp` `branch_and_bound`, `global_fix` | Beasley 1987; 열 집합을 줄여 올리는 구성은 sifting(Bixby et al. 1992)과 같은 생각 |
| reliability branching(의사비용, 신뢰도 미달 변수는 strong branching, 곱 점수) | `bnb.cpp` `branch_var` | Achterberg, Koch & Martin 2005 |
| Ryan–Foster 분기(두 행을 함께 덮는 양이 분수인 쌍) | `bnb.cpp` `branch_pair` | Ryan & Foster 1981 |
| feasibility pump(반올림점과 L1 거리가 가장 가까운 LP 점을 번갈아 구함, 정체 시 뒤집기·흔들기) | `bnb.cpp` `feasibility_pump` | Fischetti, Glover & Lodi 2005 |
| LP diving(값이 큰 변수를 1로 고정, 불가능하면 마지막 고정을 뒤집음) + 무작위 재시작 | `bnb.cpp` `dive_for_incumbent` | diving은 MIP 솔버의 일반적 휴리스틱. 재시작·뒤집기 한도는 자체 구성 |
| exact cover 탐색(쓸 수 있는 열이 가장 적은 행부터, 선택한 열과 겹치는 열 금지, 한도를 늘리며 재시작) | `bnb.cpp` `cover_search` | 행 선택 규칙은 Knuth 2000 "Dancing links"의 Algorithm X(서지 미대조). 열 순서(루트 reduced cost)와 재시작은 자체 구성 |
| LP 해 greedy 반올림(충족 안 된 행을 돕고 상한을 넘지 않는 열을 LP 값 순으로, 뒤에 중복 열 제거) | `bnb.cpp` `round_greedy` | 자체 구성 |
| SC 풀이에서 스레드 하나가 루트 LP와 트리 탐색을 맡고, 루트 LP의 reduced cost를 휴리스틱 worker의 열 제거에 씀 | `src/core/sc_solver.cpp` `run_hybrid`, `apply_rc_fix` | 자체 구성(Beasley 1987의 reduced cost fixing을 LP 값으로 적용) |
| 트리 스레드 반환: 루트 LP 시간 한도, 남은 시간에 풀 수 있는 노드 수 추정, 하한이 오르는 최근 속도로 남은 시간을 추정해 가망이 없으면 그 스레드를 휴리스틱 worker로 전환 | `sc_solver.cpp` `run_hybrid`(`H.keep`) | 자체 구성 |
| 큐에서 꺼낸 노드를 부모의 최적 기저에서 시작, strong branching 뒤 기저와 dual steepest edge 가중치를 함께 복원 | `bnb.cpp` `node_basis`, `lp.cpp` `LpSolver::save_state` | 노드 LP warm start는 MIP 솔버의 일반적 방법. 가중치 복원은 Koberstein 2005의 가중치 정의(기저에만 의존)에서 나온 자체 구성 |
| 노드별 reduced cost fixing(노드 LP 값 + reduced cost가 기준을 넘는 열을 그 가지에서 고정) | `bnb.cpp` `local_fix` | Beasley 1987의 reduced cost fixing을 노드 LP에 적용(MIP 솔버의 일반적 방법) |
| strong branching 조기 중단(연속 8개 후보가 최고 점수를 못 넘으면 멈춤) | `bnb.cpp` `branch_var` | Achterberg, Koch & Martin 2005의 lookahead |
| LP에 행 추가(새 행 변수를 기저에 넣어 dual simplex로 이어 풂)·느슨한 행 삭제 | `lp.cpp` `LpSolver::add_rows`, `remove_rows` | dual simplex의 표준 재최적화. simplex 객체를 다시 만들고 기저만 넘기는 구현은 자체 구성 |
| zero-half 컷: 홀수 개의 행을 더해 2로 나누고 올림(덮기 쪽) 또는 내림(묶기 쪽 — odd hole 부등식이 여기에 속함). 분리는 greedy(행을 하나씩 더하며 "여유 합 + 홀수 번 덮는 열의 LP 값 합"이 1 미만인 홀수 집합을 찾음) | `src/core/cuts.cpp` `separate_zero_half` | Caprara & Fischetti 1996(정의와 위반 조건); odd hole은 Hoffman & Padberg 1993. greedy 분리는 자체 구성 |
| clique 컷: 묶기 행을 공유하는 열들의 clique를 LP 값이 큰 열부터 greedy로 만들고 0인 열로 확장 | `cuts.cpp` `separate_cliques` | Hoffman & Padberg 1993 |
| 루트 컷 라운드(효율 순 선택, 느슨한 컷 삭제, 컷 뒤의 LP 값과 reduced cost로 다시 열 고정, 밀도 한도, 효과 없으면 전부 제거) | `bnb.cpp` `root_cuts` | cut-and-branch의 일반적 구성. 한도 값들은 자체 구성 |
| RINS: 노드 LP 해와 최고해가 일치하는 열을 고정하고 나머지를 작은 문제로 풂 | `bnb.cpp` `Search::rins`, `bin_solver.cpp` `lns_step` | Danna, Rothberg & Le Pape 2005 |
| 고정 이웃 탐색: 최고해의 무작위 일부를 1로 고정하고, 나머지 최고해 열 + 루트 reduced cost가 작은 열만으로 부분 문제를 풂(크기 자동 조절). 일반 경로에서 남는 스레드가 맡음 | `bin_solver.cpp` `lns_step` | large neighbourhood search(Shaw 1998)의 일반적 틀. 이웃 정의와 조절 규칙은 자체 구성 |
| 분할 문제 presolve: 중복 열, 분할 행의 포함 관계(큰 행의 나머지 열 = 0), 열이 하나 남은 행 고정 | `bin_solver.cpp` `presolve_bin` | Borndörfer 1998(학위논문 2장의 SPP 전처리) |
| 제한 트리에서 해를 찾으면 바로 정확한 트리로 전환 | `bnb.cpp` `Search::run`(`SWITCH`) | 자체 구성 |
| 시작해 생성(벤치마크용): 최선해 주변 destroy/repair("근처"), 빈 해에서 greedy + 하강("독립") | `python/scsp/bench/warm.py` | ALNS(Ropke & Pisinger 2006)의 destroy/repair를 흉내 낸 자체 구성 |

솔버 기법을 구현할 때마다 이 표에 한 줄씩 추가한다(P4분까지 반영, 2026-11-19).

P2에서 새로 인용한 문헌 두 건은 Crossref 대조를 하지 않았다(네트워크 조회를 하지 않고 기억한 서지로 적음 — 확인 필요).

- Harris, P.M.J. (1973). Pivot selection methods of the Devex LP code. *Math. Programming* 5(1), 1–28.
- Markowitz, H.M. (1957). The elimination form of the inverse and its application to linear programming. *Management Science* 3(3), 255–269.

P3에서 새로 인용한 문헌 두 건도 대조하지 않았다(네트워크 조회 없음 — 확인 필요).

- Land, A.H., Doig, A.G. (1960). An automatic method of solving discrete programming problems. *Econometrica* 28(3), 497–520.
- Knuth, D.E. (2000). Dancing links. In *Millennial Perspectives in Computer Science*, 187–214. (arXiv:cs/0011047)

P4에서 새로 인용한 문헌 두 건도 대조하지 않았다(네트워크 조회 없음 — 확인 필요). P4의 나머지 출처(Caprara & Fischetti 1996,
Hoffman & Padberg 1993, Danna et al. 2005, Borndörfer 1998, Achterberg, Koch & Martin 2005)는 1장에서 이미 대조한 것이다.

- Shaw, P. (1998). Using constraint programming and local search methods to solve vehicle routing problems. In *CP-98*, LNCS 1520, 417–431.
- Ropke, S., Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. *Transportation Science* 40(4), 455–472.
