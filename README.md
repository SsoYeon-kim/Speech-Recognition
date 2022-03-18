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
