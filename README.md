# Speech-Recognition   
   
음성인식 기본개념   
   
## 1. Waveform   
   
- 음성 데이터는 waveform파일로 저장됨
- 전처리를 통해서 발음의 종류, 성별, 음색 높이 등을 추출할 수 있음
- 푸리에 변환(FT)를 사용해 특정 시간 길이의 frame이 각각의 주파수 성분들을 얼만큼 갖고 있는지를 의미하는 spectrum을 얻을 수 있음
- 음성 전체로부터 얻은 여러 개의 spectrum을 시간 축에 나열하면 시간 변화에 따른 Spectrum의 변화인 spectrogram을 얻게 됨
