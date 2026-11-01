# 004. 모델 실험 및 오류 분석

**프로젝트:** 온라인 쇼핑 구매 여부 예측 (분류)  
**작성 범위:** CP-04~CP-08

> Notebook에서 확인한 실제 성능과 관찰을 작성합니다. 각 항목의 `___`를 실제 값이나 1~2문장으로 바꾸세요. Precision·Recall·F1은 모두 구매 클래스(`Revenue=1`) 기준입니다.

## 1. 모델 평가 기준 (CP-04)

- **양성 클래스:** 구매 (`Revenue=1`)
- **주 평가 지표:** F1 (구매 클래스)
- **보조 지표:** Precision, Recall, Accuracy
- **모델 비교에 사용한 데이터:** ___
- **Accuracy가 아니라 F1을 주요 지표로 선택한 이유:** ___ (최종 판단은 `adr.md` 의사결정 ③에 작성)

## 2. 모델별 실험 결과 (CP-04~CP-06)

대표 설정별 Train F1 / Validation F1과 Validation Precision·Recall을 기록합니다. 전체 실험 결과는 Notebook의 `results_df`와 `results/02_model_comparison.csv`에서 확인합니다.

- **DummyClassifier (most_frequent):** Train F1 ___ / Validation F1 ___ (Validation Accuracy ___)
- **LogisticRegression:** Train F1 ___ / Validation F1 ___ (Precision ___ / Recall ___)
- **DecisionTreeClassifier (`max_depth=___`):** Train F1 ___ / Validation F1 ___ (Precision ___ / Recall ___)
- **RandomForestClassifier (설정 ___):** Train F1 ___ / Validation F1 ___ (Precision ___ / Recall ___)
- **LogisticRegression의 Validation 혼동행렬:** TN ___ / FP ___ / FN ___ / TP ___

**실험에서 발견한 점**

- **기준 모델의 Accuracy와 F1이 크게 다른 이유:** ___
- **Decision Tree 깊이 변화에 따른 성능(과대적합이 나타난 깊이):** ___
- **Random Forest 두 설정의 차이:** ___
- **모델 간 Accuracy 차이와 F1 차이 비교에서 관찰한 현상:** ___ (`003_data_analysis`의 1절 클래스 비율과 연결해 작성)

## 3. 예측 오류 및 분류 임계값 분석 (CP-07)

- **비교한 후보 모델 2개(Validation F1 상위 2개):** ___ / ___
- **임계값 0.5에서 후보별 FP / FN:** ___ / ___ , ___ / ___
- **임계값을 0.3으로 낮췄을 때 변화(Precision·Recall, FP·FN):** ___
- **임계값을 0.7로 높였을 때 변화(Precision·Recall, FP·FN):** ___
- **F1이 가장 높은 모델·임계값 조합:** ___
- **모델이 상대적으로 잘 예측하지 못하는 상황(FN이 남는 이유):** ___
- **(선택) PR 곡선 또는 `class_weight` 실험에서 확인한 점:** ___

**분석:** FP(미구매를 구매로)와 FN(구매를 놓침) 중 이번 문제에서 어느 쪽 오류가 더 비싼지, 그 판단이 임계값 선택에 어떤 영향을 주었는지 작성합니다.  
___

## 4. 최종 모델과 Test 평가 (CP-08)

- **Test 확인 전에 선택한 최종 모델·설정·임계값:** ___
- **재학습에 사용한 데이터(Train + Validation) 크기:** ___행
- **Test Accuracy:** ___
- **Test Precision:** ___
- **Test Recall:** ___
- **Test F1:** ___
- **Test 혼동행렬:** TN ___ / FP ___ / FN ___ / TP ___
- **모델 저장·재로딩, 설정 파일(`02_model_config.json`) 저장 및 예측 함수 실행 결과:** ___

**해석:** 최종 Test 결과를 같은 모델·임계값의 Validation 결과와 비교해 알게 된 점을 1~2문장으로 작성합니다.  
___

> Test는 최종 평가용입니다. Test 결과를 본 뒤 모델이나 임계값을 다시 고르지 않습니다.

## 5. 실험 요약

- **가장 중요한 실험 결과:** ___
- **남아 있는 모델의 한계:** ___

> 최종 모델과 임계값을 **왜 선택했는지**는 `adr.md`의 의사결정 ④에 작성합니다.
