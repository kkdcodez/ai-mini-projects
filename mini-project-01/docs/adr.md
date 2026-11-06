# ADR — 핵심 의사결정 기록

**프로젝트:** 모듈 1 ML/DL 기초 미니프로젝트 — 회귀·분류

> 실험 결과를 반복해서 옮겨 적지 않습니다. **선택 → 핵심 근거 → 한계**를 중심으로 각 항목을 1~2문장으로 작성합니다.  
> 이 파일은 프로젝트 전체의 의사결정을 순서대로 쌓는 기록입니다. 1일차(회귀)는 ①·②, 2일차(분류)는 ③·④를 이어서 작성합니다.

---

## 1일차 — 회귀 (공유자전거 대여 수요 예측)

### 의사결정 ① 입력 특성 선택 (CP-02)

**문제**

예측 시점에 사용할 수 없거나 누수가 발생할 수 있는 특성이 있습니다.

- **검토한 선택지:** `casual`·`registered` 포함 여부, `atemp` 유지 여부, 상관이 낮은 `windspeed` 제거 여부, `instant`·`dteday`의 입력 사용 여부
- **최종 선택(사용·제외한 특성):** `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `hum`, `windspeed` **11개를 사용**하고, CP-03에서 `is_commute_hour`를 추가했다. `casual`, `registered`, `atemp`, `instant`, `dteday`는 제외했다 (`cnt`는 타깃).
- **선택 근거(누수 확인, 예측 시점, 상관관계 등):** 전 행에서 **`casual + registered == cnt`**가 성립해 두 변수를 제외했다. `temp`와 `atemp`의 상관계수가 **0.992**여서 `temp`만 사용했고, `windspeed`는 `cnt`와의 상관이 **0.094**로 낮아도 비선형 관계 가능성을 확인하기 위해 유지했다.
- **감수한 한계 또는 추후 확인할 점:** `windspeed`의 실제 예측 기여는 선형 상관계수만으로 판단할 수 없다. CP-07의 Validation permutation importance는 **0.0**으로 나타났으므로, 필요하다면 특성 제외 전후의 재학습·검증으로 효용을 추가 확인할 수 있다.

**분석 근거:** `01_regression_data_analysis.md` (CP-02~CP-03), `01_regression_model_analysis.md` (특성 중요도)

### 의사결정 ② 최종 회귀 모델 선택 (CP-07)

**문제**

서로 다른 회귀 모델 및 설정 중 하나를 최종 모델로 선택해야 합니다.

- **비교한 모델·설정:** Dummy(mean), LinearRegression, DecisionTree(`max_depth=3/6/12`), RandomForest A(`n_estimators=50`, `max_depth=10`)·B(`n_estimators=100`, `max_depth=15`). CP-07에서는 RF-B와 Tree-12의 시간대별 오류도 비교했다.
- **최종 선택한 모델·설정:** **RandomForest B** — `n_estimators=100`, `max_depth=15`, `random_state=42`, `n_jobs=-1`
- **선택 근거(Validation MAE, 과대적합 가능성, 시간대별 오류 등):** RF-B의 **Validation MAE 64.517대**는 RF-A(**76.379대**)·Tree-12(**79.959대**)보다 낮았다. 오류 분석에서도 Tree-12보다 **24개 중 23개 시간대**에서 MAE가 낮아 최종 모델로 선택했다.
- **선택 모델의 장점:** 여러 트리의 예측을 평균하며 비선형 수요 패턴을 반영했고, 비교 대상 Tree-12보다 잔차 표준편차도 낮았다(**91.3대 vs 115.3대**).
- **남아 있는 한계 또는 개선 방향:** RF-B의 Train·Validation MAE는 각각 **22.255대·64.517대**이며, 특히 **17~18시 Validation MAE가 120.3대·112.7대**로 크다. 시간 순서에 따른 분포 차이와 피크 수요의 과소예측을 추가 검증할 필요가 있다.

**분석 근거:** `01_regression_model_analysis.md`

---

## 2일차 — 분류 (온라인 쇼핑 구매 여부 예측 · PyTorch MLP)

> **아래는 2일차 작성용 양식입니다.** 이번에 제공된 완성 Notebook은 1일차 회귀만 다루므로, 분류 실험 결과는 임의로 채우지 않았습니다.

### 의사결정 ③ 평가 지표 선택 (CP-04)

**문제**

구매 세션이 전체의 약 15%뿐인 불균형 데이터이므로, Accuracy만으로 구매 클래스를 얼마나 잘 예측하는지 판단하기 어렵습니다.

- **검토한 선택지:** ___ (예: Accuracy, 구매 클래스 F1, Precision, Recall)
- **최종 선택한 주 지표와 보조 지표:** ___
- **선택 근거(DummyClassifier와 LogisticRegression 결과, FP·FN 등):** ___
- **감수한 한계 또는 추후 확인할 점:** ___

**분석 근거:** `02_classification_data_analysis.md`, `02_classification_model_analysis.md`

### 의사결정 ④ PyTorch MLP 설정 및 분류 임계값 선택 (CP-07)

**문제**

기본 손실 함수로 학습한 MLP-A와 클래스 가중 손실을 적용한 MLP-B를 비교하고, 구매 확률을 0/1로 구분하는 임계값(0.3·0.5·0.7)을 선택해야 합니다.

- **검토한 선택지:** ___ (MLP-A / MLP-B의 학습 결과, 임계값 0.3·0.5·0.7)
- **최종 선택한 MLP 설정·임계값·재학습 에포크:** ___
- **선택 근거(Validation F1, Precision·Recall, FP·FN 및 가중 손실의 영향):** ___
- **LogisticRegression과 비교했을 때 선택한 MLP의 특징:** ___
- **감수한 한계 또는 추후 개선 방향:** ___

**분석 근거:** `02_classification_model_analysis.md`
