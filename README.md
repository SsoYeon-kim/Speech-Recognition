# Speech-Recognition   
   
음성인식 기본개념   
   
## 1. Waveform   
   
- 음성 데이터는 waveform파일로 저장됨
- 전처리를 통해서 발음의 종류, 성별, 음색 높이 등을 추출할 수 있음
- 푸리에 변환(FT)를 사용해 특정 시간 길이의 frame이 각각의 주파수 성분들을 얼만큼 갖고 있는지를 의미하는 spectrum을 얻을 수 있음
- 음성 전체로부터 얻은 여러 개의 spectrum을 시간 축에 나열하면 시간 변화에 따른 Spectrum의 변화인 spectrogram을 얻게 됨   
   
## 2. Spectrogram   
   
- 전체 waveform에 한 번에 FT를 할 경우 time domain이 사라져 해석에 어려움이 생김
   - STFT를 사용
   - 자른 frame마다 FT를 취해 각각을 시간 순으로 옆으로 쌓아 time domain을 살리기 위한 방법

1. 자른 frame에 STFT를 취하면 x축이 time에서 frequency로 변함 (이를 spectrum)
2. y축에 magnitude를 제곱해준 것을 power라고 정의 (이를 power spectrum)
3. 일반적으로 magnitude에 log scale을 한 데시벨(dB)단위를 많이 사용 (이를 log-spectrum)
4. log-spectrum을 구하고 이를 세로로 세워 frame마다 옆으로 쌓으면 Spectrogram
+ x축이 각 frame이므로 time의 의미를 가진다고 해석
+ y축이 frequency, dB이 z축으로 색으로 표현됨
   
- 사람이 음성신호를 인식할 때는 주파수에 따라 선형적으로 인식하지 않고 Mel-scale로 인식
   - waveform을 Mel-scale로 변환한 뒤 spectrogram으로 변환한 것을 Mel-spectrogram이라고 함
   
## 3. MFCC(Mel-Frequency Cepstral Coefficient)   
   
- 오디오 데이터를 그대로 사용하기 보다 신호의 성질을 잘 반영하는 feature를 뽑는 것이 좋음
   - 소리의 고유한 특징을 나타내는 수치
- 가장 대표적인 오디오 feature를 뽑는 과정
- Mel spectrum에서 Cepstral 분석을 통해 추출된 값   
   
1. 전체 오디오 신호를 일정 간격으로 나누고 FT를 거쳐 spectrogram을 구함
2. 각 spectrum의 제곱인 power spectrogram에 Mel filter bank를 적용해 Mel Spectrum을 구함
3. cepstral 분석을 적용해 MFCC를 구함   
   
* cepstral 분석 : spectrum에서 배음 구조를 유추해낼 수 있다면 소리의 고유한 특징을 찾을 수 있음, 이것을 가능하게 하는 것 cepstral분석   
* 배음(harmonics) : 소리에는 기본 주파수와 함께 기본 주파수의 정수배인 배음들로 구성   
   
Ex) MFCC는 formants을 연결한 곡선과 spectrum을 분리하는 과정에서 도출됨   
   
<img src="https://user-images.githubusercontent.com/62587484/158983479-846c3671-8b93-43fb-b616-cf8df053f0ed.png" width="100%">   
   
* formants : 소리가 공명되는 특정 주파수 대역, 배음과 만나 소리를 풍성하게 혹은 선명하게 만드는 필터 역할   
   
## 4. HMM(Hidden Markov Model)   
   
- 순차데이터를 확률적으로 모델링하는 생성 모델
- 같은 시간에 발생한 두 종류의 state sequence 각각의 특성과 관계를 이용해 모델링
   - hidden state sequence (관측 불가능) : S
   - observable state sequence (관측 가능) : O
   
Markov 가정 : 시간 t에서의 관측은 가장 최근 r개의 관측에만 의존한다 | 미래의 상태는 오직 현재의 상태의 영향만 받는다.   
ex) if r=1, p(St|St-1) : first order markov model , if r=2, p(St|St-1, St-2) : second order markov model   

- S(hidden sequence)는 Markov 가정을 따름 (순차적 특성을 반영)
- O(observable sequence)는 S에 종속

#### [HMM의 파라미터] : λ = (A, B, π)   
   
- A(aij) : 상태전이확률 행렬 (state transition probability matrix)
- B(bjk) : 방출확률 행렬 (S에서 O를 볼 확률)  (Emission probability matrix)
- π(πi) : 각 hidden state의 초기 확률 (initial state probability matrix)
   
1. Evaluation : HMM(λ*), O가 주어졌을 때 O의 확률은 무엇인가
2. Decoding : HMM(λ*), O가 주어졌을 때, S(hidden state sequence)를 알아내는 문제
3. Learning : O가 주어졌을 때(데이터가 주어졌을 때), HMM파라미터를 추정 (parameter estimation)
    
O(training sequence data)가 있을 때 3. Learning을 통해 모델을 만들고   
O(testing sequence data)를 만들어 놓은 모델을 통해   
1. Evaluation, 2. Decoding 문제를 해결할 수 있음   
   
#### [Evaluation]   
   
- S의 개수와 sequence의 길이가 많아지면 경우의 수가 늘어가 풀기 복잡해짐
   - solution : Forward algorithm (시간순으로 확률 계산 ->)		: α
   - Backward algorithm (<- 방향으로 확률 계산) 		: β
   - Probability_Forward(O) = Probability_Backward(O)   

<img src="https://user-images.githubusercontent.com/62587484/158984177-2aff66f0-d900-44cc-b34a-ed235a54d843.png" width="50%">   
   
#### [Decoding]   
   
- solution : Viterbi algorithm

<img src="https://user-images.githubusercontent.com/62587484/158984315-7cf3f7ab-74b8-4134-8dd7-d34e413700ce.png" width="70%"><img src="https://user-images.githubusercontent.com/62587484/158984341-529b3ebf-97a1-4181-8002-170042c3547b.png" width="30%">   
   
Forward algorithm for evaluation : 가능한 모든 경우의 확률의 합   
Viterbi algorithm for decoding : 가능한 모든 경우의 확률의 최대   
   
(forward에서는 O의 확률을 구하기 때문에 sum, viterbi에서는 관측치에서 가장 그럴싸한 S를 구하기 때문에 max)   
   
#### [Learning]   
   
- 관측 벡터 O의 확률을 최대로 하는 HMM(λ)를 찾는 것
   - HMM(λ*) = argmax P(O | λ)   

1. HMM 초기화   
	(A, B, π -> γ, ξ -> A*, B*, π* -> γ*, ξ* -> A**, B**, π** -> γ**, ξ**)   
2. 적절한 방법으로 P(HMM(λnew)) > P(O | HMM(λ))를 찾기
3. 만족스러우면 λ^ = HMM(λnew)으로 설정하고 멈춤, 혹은 2번 반복

Solution : Baum-Welch algorithm (= forward-backward algorithm)
- γt(i) : HMM(λ), O가 주어졌을 때, t시점 상태가 si(hidden state가 i)일 확률
- ξt(i,j) : HMM(λ), O가 주어졌을 때, t시점 상태가 si, t+1시점 상태는 sj일 확률
   
- 두 가지 step을 번갈아 가며 진행 (E-M 알고리즘)
   - E-step : α, β를 계산해서 γt(i), ξt(i,j)를 구하는것
   - M-step : γt(i), ξt(i,j)를 이용해 HMM(λ)를 개선 -> HMM(λnew)
      -  개선 : P(HMM(λnew)) > P(O | HMM(λ))   즉, HMM(λnew)의 evaluation시 높은 확률   
   
#### [음향모델에서의 사용]   
   
- 음향모델에서의 사용
   - 음소의 연쇄가 드러나지 않아 오디오 파형같이 보이는 대체물을 이용해 간접적으로 훈련
   - 음성신호로부터 추출된 특징벡터의 나열을 음소의 나열로 바꿔주는 역할
   
* 음소의 연쇄 : 동일한 음소의 연쇄(복잡한 파형을 갖는 음소를 하나의 상태만으로 표현하기엔 부족함)   
   
Ex) ‘t’ 파형의 처음, 중간, 마지막 상태로 음소의 동일성은 유지하되, 구간을 나눠 음소의 복잡한 특징을 상태별로 담을 수 있도록 마르코프 체인을 구성하여 학습을 진행.
이때, 학습하고자 하는 음소의 상태는 hidden영역, 볼 수 있는 오디오는 observation 영역   
   
## 5.GMM(Gaussian Mixture Model)   
   
- clustering 방법 중 하나로 데이터의 군집을 가우시안 모델로 표현하는 기법
- 가우시안 모델의 평균(μ)과 분산(σ^2)으로부터 군집의 특성을 알 수 있음

#### [Mixture Model]
- 전체 분포에서 하위 분포가 존재한다고 보는 모델
- 데이터가 모수를 갖는 여러 개의 분포로부터 생성되었다고 가정하는 모델
   
가장 많이 사용되는 GMM은 데이터가 k개의 정규분포로부터 생성되었다고 보는 모델   
   
Ex) A, B, C 세 종류의 정규분포의 PDF(확률밀도함수)를 보면 GMM은 세 가지 정규분포 중 하나로부터 생성되었으며, 그 안에서 random하게 정규분포 형태를 갖고 생성되었다고 보는 모델   

<img src="https://user-images.githubusercontent.com/62587484/158985051-6c2ba676-eea1-400b-82a7-0dc9ab2370a6.png" width="50%"><img src="https://user-images.githubusercontent.com/62587484/158985082-27664e21-3123-406d-95d1-aed5256e760f.png" width="50%">   
   
위 사진을 보고
- 3개의 정규분포의 평균과 분산을 어느정도 짐작할 수 있음
- 각각의 분포로부터 생성되는 데이터의 양이 차이가 있다는 것을 알 수 있음   
=> 3개의 분포의 평균과 모수, Weight | 총 9개의 모수를 통해 데이터의 분포를 설명할 수 있는 혼합 PDF를 만들어낼 수 있음
    
#### [모수 추정]
- 세 가지 정규분포 중 확률적으로 어디에 속해 있는가를 나타내는 Weight값 : 잠재변수
- 각각의 정규분포의 모수(평균, 분산)   
   
[MLE(Maximum Likelihood Estimation, 최대우도법)]
- 파라미터 θ=(θ1,⋯,θm)으로 구성된 어떤 확률밀도함수 P(x | θ)에서 관측된 표본 데이터 집합을 x=(x1,x2,⋯,xn)이라고 할 때, 이 표본들에서 파라미터 θ=(θ1,⋯,θm)를 추정하는 방법   
   
<img src="https://user-images.githubusercontent.com/62587484/158985287-f6ef7f92-9431-48b5-9526-c46fbe4b056e.png" width="70%">   
   
위 사진에서 데이터를 관찰함으로써 이 데이터가 추출되었을 것으로 생각되는 분포의 특성을 추정할 수 있음   
(추출된 분포가 정규분포라 가정, 분포의 특성 중 평균을 추정하려 함)   
   
*likelihood : 지금 얻은 데이터가 이 분포로부터 나왔을 가능도, 각 데이터 샘플에서 후보 분포에 대한 높이   
   
Likelihood function : 전체 표본집합의 결합확률밀도함수   

<img src="https://user-images.githubusercontent.com/62587484/158985465-9c78cb3a-cd7f-4d3f-b55f-d134ba0e8466.png" width="50%">   
   
보통, 자연로그를 이용해 log-likelihood function L(θ | x)를 이용   

<img src="https://user-images.githubusercontent.com/62587484/158985552-f4985e8d-0f05-4689-b30c-02c298f92067.png" width="50%">   
   
MLE는 likelihood 함수의 최대값을 찾는 방법   
- 미분계수를 이용해 최대값을 구함   
   
- 데이터에 대해 정규분포를 가정했을 때, 주어진 m개의 데이터에 대한 두 파라미터(평균, 분산)에 대한 MLE는 주어진 데이터를 {x(1), x(2), ..., x(m)}이라고 했을 때 평균, 분산 값을 계산할 때 likelihood 함수가 최대가 된다.   
(아래 수식은 주어진 데이터에 대한 likelihood를 계산한 후, likelihood가 최대가 되게 하는 평균과 분산 값을 얻은 결과)   

<img src="https://user-images.githubusercontent.com/62587484/158985729-c1664e79-403b-469c-9233-afb4f535ca06.png" width="50%">   
   
아래 사진 처럼 데이터 라벨이 주어지지 않은 경우
- 랜덤하게 라벨 or 분포를 설정해주고 시작 (여기서는 각 라벨이 해당하는 분포를 랜덤하게 주고 시작)
   
<img src="https://user-images.githubusercontent.com/62587484/158985784-0ecf3161-eb4a-4b28-ac51-5542c082a206.png" width="50%">  
   
아래의 방식으로 모든 데이터 샘플들에 대한 라벨을 확인하면 분포는 수렴함
1. 랜덤하게 분포 제안
2. likelihood 비교로 라벨링
3. 각 그룹별 모수 추정
4. 추정된 모수를 이용한 각 그룹별 분포 도시
   
<img src="https://user-images.githubusercontent.com/62587484/158985956-29d773fb-64fc-41c8-92d7-4848efc0b568.png" width="50%">  
   
=> 라벨이 주어지지 않은 데이터에 대해 데이터셋은 정규분포를 이룰 것이라 가정하고 클러스터링을 수행해주는 과정을 GMM이라고 함
   
#### E-M알고리즘(Expectation-Maximization)]
- 두 가지 step을 번갈아 가며 진행
	- E-step : 각 데이터에 라벨을 부여하는 과정
	- M-step : 각 그룹의 모수를 재 계산하는 과정 (평균, 표준편차 계산)
   
## 6. DNN-HMM   
   
- GMM모델을 DNN으로 대체한 시스템
- 입력으로 주어지는 특징 벡터에 대해 발음 사전에 정의된 음소 중, 가장 높은 확률을 가지는
음소를 출력하는 분류과정을 수행

1. HMM 학습 과정을 통해 DNN 구조의 출력 노드 또는 타겟에 해당하는 HMM의 상태를 결정
2. 훈련 음성 데이터의 상태 레벨 정렬 정보(state-level alignment)를 추출

- DNN의 학습 과정
	- HMM 학습 결과로 이미 결정되는 상태(state) 및 훈련 음성 데이터의 상태 레벨 정렬 정보를 전달받아, 단순히 패턴 인식 측면에서 가장 변별력이 있는 형태의 특징 및 모델 파라미터를 얻는 과정
   
<img src="https://user-images.githubusercontent.com/62587484/158986281-e5a9126e-ad48-41e1-b947-b4d7ac2fa0dd.png" width="80%">   
   
## 7. TDNN(Time Delay Neural Network)   
   
- input 데이터로 시간에 대해 (t, t-1, t-2)인 데이터를 한 번에 넣는 방법(시간을 고려한 모델이 아닌 데에 discrete한 시간 데이터를 다룰 때 쓰는 대표적인 방법)
- 연속적인 시간 데이터의 경우는 위 방법을 사용할 수 없음
	-  sliding window 방식으로 일정한 길이의 데이터를 일정 부분 겹치도록 전처리한 뒤 사용
- 음성과 같은 동적인 패턴을 다루기 위해 지연요소, 시간적으로 처리하는 요소 등을 이용해 입력 패턴에 내재되어 있는 시간적인 특징을 인식하는 신경망
- 시간 차원을 따라 가중치를 공유하는 네트워크
   
<img src="https://user-images.githubusercontent.com/62587484/158986443-8b7ca260-6bbb-48c3-9ccc-b1de651c9050.png" width="50%">  
   
- 학습 알고리즘
	- 오류역전파(back propagation)학습 알고리즘 이용 (경사 하강법을 사용해 MSE가 최소가 되도록 연결강도를 조정하는 학습법)

1. 각 time-shifted window에 대한 weight들은 서로 같도록 제한
2. 우선 각각의 time-shifted window들에 대해 BP알고리즘을 적용해 weights의 변화량을 계산
3. 그 변화량들의 평균값으로 각 weight를 조정
   
아래 사진은 ‘ㅂ’, ‘ㄷ’, ‘ㄱ’ 세 개의 음소를 인식하는 TDNN의 전체 구조를 나타냄   
   
<img src="https://user-images.githubusercontent.com/62587484/158986536-4ac34775-3d0c-45c1-beed-2629cdfd2215.png" width="50%">   
   
- 두 은닉층의 각 노드 열은 앞 노드 열을 그대로 복제한 형상
- 은닉층에서 같은 행에 위치한 노드들은 동일한 weight 집합을 가지게 됨 (= 한 행에서 서로 대응되는 위치에 놓인 connection들은 같은 weight를 가짐)
- 하나의 TDNN으로 음소 전체를 인식케 하는 것은 학습에 소요되는 시간이나 질을 따져볼 때
바람직하지 않음 (음소를 여러 그룹으로 묶고 그 그룹 수만큼 TDNN을 둬 각 TDNN이 담당 그룹만을 전문적으로 인식케 하는 다중 인식기 구조가 흔히 사용됨)

