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

<img src="https://user-images.githubusercontent.com/62587484/158984177-2aff66f0-d900-44cc-b34a-ed235a54d843.png" width="100%">   
   
#### [Decoding]   
   
- solution : Viterbi algorithm

<img src="https://user-images.githubusercontent.com/62587484/158984315-7cf3f7ab-74b8-4134-8dd7-d34e413700ce.png" width="50%"><img src="https://user-images.githubusercontent.com/62587484/158984341-529b3ebf-97a1-4181-8002-170042c3547b.png" width="50%">   
   
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
   
#### 음향모델에서의 사용   
   
- 음향모델에서의 사용
   - 음소의 연쇄가 드러나지 않아 오디오 파형같이 보이는 대체물을 이용해 간접적으로 훈련
   - 음성신호로부터 추출된 특징벡터의 나열을 음소의 나열로 바꿔주는 역할
   
* 음소의 연쇄 : 동일한 음소의 연쇄(복잡한 파형을 갖는 음소를 하나의 상태만으로 표현하기엔 부족함)   
   
Ex) ‘t’ 파형의 처음, 중간, 마지막 상태로 음소의 동일성은 유지하되, 구간을 나눠 음소의 복잡한 특징을 상태별로 담을 수 있도록 마르코프 체인을 구성하여 학습을 진행.
이때, 학습하고자 하는 음소의 상태는 hidden영역, 볼 수 있는 오디오는 observation 영역   
   
## GMM(Gaussian Mixture Model)   
   


