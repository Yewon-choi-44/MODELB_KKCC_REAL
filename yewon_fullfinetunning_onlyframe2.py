"""
# 아키텍처 변형형 풀 파인튜닝
    최종 역할
    - 원래 모델이 하던 일(텍스트 문맥 이해)을 버리고
    - 내가 붙여놓은 새로운 작업(3가지 카테고리 분류)을 수행하는 완전히 다른 도구로 개조.

    컴퓨터 내부의 동작:
    - 기존 베이스 모델의 출력단 위치에
        nn.Linear(256, num_labels)라는
        완전히 새로운 계산 장치(레이어)의 설계도를
        프로그래머가 코드로 강제로 이어 붙인 상태.
    - 이로 인해 모델의 최종 출력 데이터 형태가 원래의 형태가 아니라,
        내가 지정한 3개의 숫자 배열(확률 점수)로 강제 변환됩니다.
    - 이 상태에서 requires_grad=True로 풀 파인튜닝을 돌리면,
        기존 베이스 모델의 내부 숫자들과 내가 새로 추가한 레이어의 숫자들이
        한 몸이 되어 동시에 오차를 수정하며 문장 분류기라는 새로운 기계로 재탄생합니다.

    손실 함수 (Loss) 기준
    - 정답 카테고리 원-핫 인코딩과의 차이 (CrossEntropy)

    최종 목적
    - 모델을 활용해 아예 새로운 작업을 하기 위함
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput


# 1. 파이토치 신경망 모듈을 상속받아 사용자 정의 모델 클래스 설계
class CustomTransformerForClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        # Hugging Face 정통 서버에서 기본 베이스 모델의 아키텍처와 가중치 로드
        self.encoder = AutoModel.from_pretrained(model_name)

        # 기본 모델의 최종 출력 차원 크기(Hidden Size)를 추출하여 커스텀 레이어에 연결
        hidden_size = self.encoder.config.hidden_size

        # [사용자 정의 추가 레이어 1]: 입력 차원을 256 차원의 압축된 크기로 변환
        self.dense_layer = nn.Linear(hidden_size, 256)
        # [사용자 정의 추가 레이어 2]: 활성화 함수로 데이터의 비선형성 확보
        self.activation = nn.ReLU()
        # [사용자 정의 추가 레이어 3]: 256 차원의 데이터를 최종 분류 클래스 개수로 변환
        self.classifier_head = nn.Linear(256, num_labels)

        # [⚠️ 중요] 풀 파인튜닝 설정을 위해 전체 파라미터의 가중치 계산 제어권을 True로 전원 개방
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        # A. 기본 임베딩 및 트랜스포머 인코더 블록 연산 수행
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # B. 문장 전체의 핵심 문맥 정보가 담긴 첫 번째 토큰([CLS])의 결과 벡터 행렬 추출
        # 추출된 행렬 크기: [배치 사이즈, Hidden Size]
        cls_representation = encoder_outputs.last_hidden_state[:, 0, :]

        # C. 내가 수동으로 추가한 커스텀 신경망 계산 레이어들을 순차 통과
        x = self.dense_layer(cls_representation)
        x = self.activation(x)
        logits = self.classifier_head(x)  # 최종 예측 점수 행렬 도출

        # D. Hugging Face Trainer와 연동하기 위해 내부에서 오차(Loss)를 직접 계산
        loss = None
        if labels is not None:
            # 분류 문제 해결을 위한 크로스 엔트로피 손실 함수 계산 적용
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits, labels)

        # E. Trainer 규격에서 요구하는 특수 출력 데이터 타입 객체에 담아서 반환
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )


# 2. 개조 완료된 나만의 모델 인스턴스 생성
model_name_or_path = "bert-base-uncased"  # 예시용 기본 베이스 모델명
my_custom_model = CustomTransformerForClassification(model_name=model_name_or_path, num_labels=3)