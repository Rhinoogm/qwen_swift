# Qwen3.5-0.8B Detector-Aware Crop Recommendation Pipeline

이 저장소는 `이미지 + detector 결과 + auto crop 결과`를 입력으로 받아, 모델이 아래 JSON을 출력하도록 학습하는 최소 파이프라인입니다.

```json
{
  "best_crop": {
    "x1": 0.1180,
    "y1": 0.0600,
    "x2": 0.9020,
    "y2": 0.9480
  },
  "reason": "The crop keeps the main person fully visible, includes the nearby salient objects, and removes empty background."
}
```

핵심은 세 가지입니다.

- detector가 이미지 안에 무엇이 어디 있는지 알려줌
- auto crop 모델이 기본 crop 후보를 하나 제안함
- 더 큰 teacher 모델이 최종 정답 crop과 이유를 만들어 줌

그 뒤 student 모델인 `Qwen/Qwen3.5-0.8B`를 `SFT -> GRPO` 순서로 학습합니다.

이 저장소는 detector나 auto crop 모델 자체를 학습하지 않습니다. 이 저장소는 그 결과물을 받아서 `crop 추천 모델`을 학습시키는 부분만 담당합니다.

## 한눈에 보기

### 입력

- 이미지 1장
- detector 결과
- auto crop top1 결과

### 출력

- 최종 crop 좌표 `best_crop`
- 그 좌표가 좋은 이유 `reason`

### 학습에 쓰는 모델 역할

- `detector`
  이미지 속 객체의 위치를 찾습니다.
  예: 사람, 강아지, 자동차 같은 객체와 bbox를 줍니다.
- `auto crop model`
  대략 괜찮은 crop을 먼저 제안합니다.
  예: `x1=0.12, y1=0.08, x2=0.92, y2=0.95`
- `teacher model`
  detector 정보와 auto crop 제안을 보고, 더 좋은 최종 crop과 이유를 생성합니다.
  이 결과가 데이터셋의 정답 `teacher_answer`가 됩니다.
- `student model`
  최종적으로 배포하고 싶은 작은 모델입니다.
  여기서는 `Qwen/Qwen3.5-0.8B`를 씁니다.

## 가장 쉬운 예시

한 장의 강아지 사진이 있다고 가정합니다.

1. detector가 이렇게 말합니다.

```json
[
  {
    "class_id": 0,
    "class_name": "dog",
    "box_cx": 0.5990,
    "box_cy": 0.5700,
    "box_w": 0.3850,
    "box_h": 0.4300
  }
]
```

2. auto crop 모델이 이렇게 제안합니다.

```json
{
  "x1": 0.3400,
  "y1": 0.1800,
  "x2": 0.9000,
  "y2": 0.9800,
  "score": 0.7800
}
```

3. teacher 모델이 이렇게 최종 정답을 만듭니다.

```json
{
  "best_crop": {
    "x1": 0.3000,
    "y1": 0.1600,
    "x2": 0.9200,
    "y2": 0.9800
  },
  "reason": "The crop keeps the running dog fully visible while trimming empty sky and preserving enough grass for context."
}
```

4. 그러면 raw annotation 한 줄은 이렇게 됩니다.

```json
{
  "sample_id": "dog-001",
  "image": "images/dog_001.jpg",
  "split": "train",
  "detector_objects": [
    {
      "class_id": 0,
      "class_name": "dog",
      "box_cx": 0.5990,
      "box_cy": 0.5700,
      "box_w": 0.3850,
      "box_h": 0.4300
    }
  ],
  "autocrop_top1": {
    "x1": 0.3400,
    "y1": 0.1800,
    "x2": 0.9000,
    "y2": 0.9800,
    "score": 0.7800
  },
  "teacher_answer": {
    "best_crop": {
      "x1": 0.3000,
      "y1": 0.1600,
      "x2": 0.9200,
      "y2": 0.9800
    },
    "reason": "The crop keeps the running dog fully visible while trimming empty sky and preserving enough grass for context."
  }
}
```

이 한 줄이 학습 데이터의 기본 단위입니다.

## 이 저장소가 하는 일

### 1. raw annotation 검증

`scripts/prepare_dataset.py`

- 이미지 경로가 실제로 존재하는지 확인
- detector 결과 형식이 맞는지 확인
- auto crop 좌표가 `0~1` 범위인지 확인
- teacher 정답 JSON이 올바른지 확인

### 2. 학습용 JSONL 생성

같은 raw 데이터에서 아래 파일을 만듭니다.

- `sft_train.jsonl`
- `sft_val.jsonl`
- `grpo_train.jsonl`
- `audit.json`

### 3. SFT

`scripts/run_sft.py`

- student 모델이 teacher 정답을 그대로 따라 하도록 먼저 학습합니다.
- 쉽게 말해 “정답 예시를 많이 보여주며 따라 쓰게 하는 단계”입니다.

### 4. merge

`scripts/merge_adapter.py`

- LoRA adapter를 실제 모델 가중치와 합칩니다.
- 다음 단계에서 편하게 쓰기 위한 중간 정리 단계입니다.

### 5. GRPO

`scripts/run_grpo.py`

- student 출력이 teacher 정답과 더 비슷해지도록 reward를 주며 추가 학습합니다.
- 이 저장소에서는 다음 reward를 씁니다.
  - IoU
  - 좌표 절대 오차
  - reason 문장 semantic similarity

### 6. 추론과 평가

- `scripts/infer.py`: 이미지 1장 테스트
- `scripts/predict_dataset.py`: 데이터셋 전체 예측
- `scripts/evaluate.py`: SFT 결과, GRPO 결과, autocrop baseline 비교

## 데이터셋을 만들 때 무엇을 준비해야 하나

실무 기준으로는 아래 순서를 추천합니다.

### 단계 1. 이미지 모으기

예:

- 상품 이미지
- 인물 사진
- SNS 업로드용 원본 이미지

이 저장소는 이미지 파일 자체는 건드리지 않습니다. 경로만 JSONL에 기록합니다.

### 단계 2. detector 돌리기

각 이미지마다 객체 목록을 만듭니다.

필수 필드:

- `class_id`
- `class_name`
- `box_cx`
- `box_cy`
- `box_w`
- `box_h`

주의:

- 좌표는 normalized `0~1`이어야 합니다.
- 픽셀 좌표를 쓰고 있다면 먼저 `image_width`, `image_height`로 나눠서 정규화하세요.

예:

```text
pixel bbox: x=120, y=40, w=360, h=520 on 600x800 image
normalized:
box_cx = (120 + 360/2) / 600
box_cy = (40 + 520/2) / 800
box_w  = 360 / 600
box_h  = 520 / 800
```

### 단계 3. auto crop 모델 돌리기

각 이미지마다 최소한 top1 crop 하나를 만듭니다.

필드:

- `x1`
- `y1`
- `x2`
- `y2`
- `score`

`score`는 현재 학습 reward에 직접 쓰이지 않지만, 나중에 분석할 때 꽤 유용합니다.

### 단계 4. detector 결과와 auto crop 결과를 하나의 raw JSONL로 합치기

이 시점에서는 `teacher_answer`가 없어도 됩니다.

예시:

```json
{
  "sample_id": "img-000001",
  "image": "images/img-000001.jpg",
  "split": "train",
  "detector_objects": [...],
  "autocrop_top1": {
    "x1": 0.12,
    "y1": 0.08,
    "x2": 0.92,
    "y2": 0.95,
    "score": 0.81
  }
}
```

### 단계 5. teacher 모델로 정답 만들기

이 저장소에서는 teacher를 학습 중에 매번 부르지 않습니다.
먼저 teacher를 오프라인으로 한 번 돌려서 `teacher_answer`를 raw JSONL에 채워 넣습니다.

즉:

- 학습 전에 teacher 정답을 미리 생성
- 학습 중에는 그 정답만 읽어서 비교

이 방식이 더 단순하고, 비용도 관리하기 쉽습니다.

실행:

```bash
python scripts/generate_teacher_labels.py \
  --model /path/to/teacher-model \
  --input data/raw/annotations.jsonl \
  --output data/raw/annotations_with_teacher.jsonl \
  --root-dir /data/my_dataset \
  --temperature 0.0
```

운영 팁:

- 이미 `teacher_answer`가 들어 있는 row는 기본적으로 재사용합니다.
- 즉, 새로 추가한 row만 채우고 싶을 때 같은 스크립트를 다시 돌려도 됩니다.
- 모든 row에 `teacher_answer`가 이미 있다면 `--model` 없이 실행해서 형식 검증과 정규화만 다시 할 수도 있습니다.
- 기존 `teacher_answer`를 전부 다시 만들고 싶을 때만 `--overwrite-existing`를 붙이세요.

이 스크립트는 아래를 자동으로 검사합니다.

- teacher 출력이 JSON인지
- `best_crop` 좌표가 유효한지
- `reason`이 비어 있지 않은지
- `reason`이 영어 한 문장인지

### 단계 6. 학습용 파일로 변환

```bash
python scripts/prepare_dataset.py \
  --input data/raw/annotations_with_teacher.jsonl \
  --output-dir data/processed \
  --root-dir /data/my_dataset
```

이제 학습에 바로 넣을 수 있는 파일이 생깁니다.

## 왜 teacher 모델이 필요한가

이 프로젝트의 강화학습은 “학생이 직접 만든 crop이 진짜로 좋은지”를 외부 crop scorer로 재채점하지 않습니다.
대신 teacher가 만든 정답과 얼마나 비슷한지를 reward로 씁니다.

즉, teacher의 역할은 두 가지입니다.

- SFT에서 정답 예시 제공
- GRPO에서 비교 기준 제공

쉽게 말하면:

- detector: 객체 위치 제공
- auto crop: 초기 제안 제공
- teacher: 최종 정답 제공
- student: 최종적으로 배포할 작은 모델

## 학습 단계가 의미하는 것

### SFT는 무엇인가

teacher 정답을 보고 그대로 따라 쓰는 연습입니다.

입력:

- 이미지
- detector_objects
- autocrop_top1

출력:

- teacher의 `best_crop`
- teacher의 `reason`

### GRPO는 무엇인가

student가 답을 생성했을 때, teacher 정답과 얼마나 비슷한지를 reward로 계산해서 더 다듬는 단계입니다.

현재 reward 식:

- `0.70 * IoU(pred_bbox, ref_bbox)`
- `0.20 * (1 - mean_abs_coord_error)`
- `0.10 * semantic_similarity(pred_reason, ref_reason)`

즉, 좌표가 가장 중요하고, reason도 약하게 반영합니다.

## 빠른 시작

### 추천 작업 디렉토리 만들기

처음에는 아래 한 번으로 작업 폴더를 맞춰 두는 편이 가장 안전합니다.

```bash
python scripts/init_workspace.py --workspace-dir .
```

이 스크립트는 아래를 준비합니다.

- `data/raw/annotations.template.jsonl`
- `data/demo_dataset/annotations.jsonl`
- `data/processed/`
- `outputs/predictions/`
- `outputs/reviews/`
- `outputs/final_answers/`

권장 사용법:

- 내 데이터셋을 만들 때는 `data/raw/annotations.template.jsonl`를 복사해서 시작
- 로컬에서 깨지지 않는 흐름을 먼저 확인할 때는 `data/demo_dataset/annotations.jsonl` 사용
- `data/demo_dataset`은 저장소에 포함된 검증용 예제를 복사한 것이므로 README 절차를 바로 따라가기 좋음

### 로컬에서 연결만 점검하기

```bash
conda create -n qwen-lora-dev python=3.10 -y
conda activate qwen-lora-dev
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .[dev]
python scripts/check_environment.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/smoke_test.py
```

이 단계는 실제 학습이 아니라 아래가 잘 연결되는지 확인합니다.

- 작업 디렉토리 초기화
- 데모 데이터셋 생성
- 데이터 검증
- SFT 명령 생성
- GRPO 명령 생성
- 평가 스크립트 동작
- 최종 답안 export

### GPU 서버에서 학습 환경 만들기

```bash
conda env create -f environment.yml
conda activate qwen-lora
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .[train,dev]
python scripts/check_environment.py --expect-train
```

주의:

- 학습은 보통 `Linux + NVIDIA GPU + CUDA` 기준입니다.
- macOS는 데이터 준비와 dry-run 확인 용도로 생각하는 편이 맞습니다.

## 실제 학습 순서

### 0. 작업 폴더 초기화

```bash
python scripts/init_workspace.py --workspace-dir .
```

### 1. 데모 데이터셋으로 먼저 한 번 검증하기

teacher 모델 없이도 데이터 준비 단계가 깨지지 않는지 먼저 확인할 수 있습니다.

```bash
python scripts/prepare_dataset.py \
  --input data/demo_dataset/annotations.jsonl \
  --output-dir data/processed \
  --root-dir data/demo_dataset
```

### 2. teacher 정답 생성

```bash
python scripts/generate_teacher_labels.py \
  --model /path/to/teacher-model \
  --input data/raw/annotations.jsonl \
  --output data/raw/annotations_with_teacher.jsonl \
  --root-dir /data/my_dataset
```

### 3. 학습용 JSONL 생성

```bash
python scripts/prepare_dataset.py \
  --input data/raw/annotations_with_teacher.jsonl \
  --output-dir data/processed \
  --root-dir /data/my_dataset
```

### 4. SFT 실행

```bash
python scripts/run_sft.py --dry-run
python scripts/run_sft.py
```

### 5. SFT adapter merge

```bash
python scripts/merge_adapter.py \
  --adapter-root outputs/sft \
  --output-dir outputs/sft_merged
```

### 6. GRPO 실행

```bash
python scripts/run_grpo.py --dry-run
python scripts/run_grpo.py
```

### 7. GRPO adapter merge

```bash
python scripts/merge_adapter.py \
  --adapter-root outputs/grpo \
  --output-dir outputs/final_merged
```

## 추론과 평가

### 단일 이미지 추론

```bash
python scripts/infer.py \
  --model outputs/final_merged \
  --image /data/my_dataset/images/img-000001.jpg \
  --detector-objects-json detector_objects.json \
  --autocrop-top1-json autocrop_top1.json
```

### 데이터셋 전체 예측

```bash
python scripts/predict_dataset.py \
  --model outputs/final_merged \
  --input data/processed/sft_val.jsonl \
  --output outputs/predictions/final_val_predictions.jsonl \
  --review-output outputs/reviews/final_val_review.jsonl
```

`--review-output`을 같이 주면 사람이 직접 수정 가능한 review JSONL도 함께 생성됩니다.
이 파일은 아래 필드를 포함합니다.

- `model_answer`: 모델이 낸 원본 구조화 답
- `final_answer`: 최종 배포용 답안. 처음에는 `model_answer`로 채워짐
- `reference_answer`: teacher 정답
- `metrics`: IoU, 좌표 오차, parse 성공 여부
- `notes`: 사람이 수정 사유를 적는 빈 칸

즉, 실제 운영에서는 `outputs/reviews/*.jsonl`만 열어서 `final_answer`를 수정하면 됩니다.

### review 파일에서 최종 답안만 내보내기

```bash
python scripts/export_final_answers.py \
  --input outputs/reviews/final_val_review.jsonl \
  --output outputs/final_answers/final_val_answers.jsonl
```

이 단계에서 `final_answer` 형식이 잘못된 row는 바로 잡아내므로, 배포 직전 검증용으로 쓰기 좋습니다.

### SFT와 GRPO 비교 평가

```bash
python scripts/predict_dataset.py \
  --model outputs/sft_merged \
  --input data/processed/sft_val.jsonl \
  --output outputs/predictions/sft_val_predictions.jsonl

python scripts/predict_dataset.py \
  --model outputs/final_merged \
  --input data/processed/sft_val.jsonl \
  --output outputs/predictions/final_val_predictions.jsonl

python scripts/evaluate.py \
  --predictions outputs/predictions/final_val_predictions.jsonl \
  --baseline-predictions outputs/predictions/sft_val_predictions.jsonl
```

`evaluate.py`는 아래를 봅니다.

- JSON parse 성공률
- bbox 유효 비율
- teacher 대비 mean IoU
- teacher 대비 좌표 오차
- reason semantic similarity
- autocrop baseline 대비 개선 여부
- SFT 대비 GRPO 개선 여부

## 자주 수정하게 되는 파일

### detector 출력 형식이 바뀌었을 때

- `src/qwen_lora/data_utils.py`

### 프롬프트 문구를 바꾸고 싶을 때

- `src/qwen_lora/prompting.py`

### 모델 출력 JSON 형식을 바꾸고 싶을 때

- `src/qwen_lora/reward_core.py`
- `src/qwen_lora/data_utils.py`
- `tests/test_reward_core.py`
- `tests/test_data_utils.py`

### reward 비중을 바꾸고 싶을 때

- `plugins/crop_reward.py`
- `configs/grpo.yaml`

### teacher label 생성 방식을 바꾸고 싶을 때

- `scripts/generate_teacher_labels.py`

### 작업 폴더와 결과물 관리를 바꾸고 싶을 때

- `scripts/init_workspace.py`
- `scripts/export_final_answers.py`

## 포함된 예시 파일

- `examples/raw_annotations.example.jsonl`
  raw JSONL 예시
- `examples/demo_dataset/annotations.jsonl`
  저장소에 포함된 검증용 작은 데이터셋
- `scripts/create_demo_dataset.py`
  포함된 데모 데이터셋을 다른 위치로 복사
- `scripts/init_workspace.py`
  `data/`, `outputs/` 기본 구조와 로컬 demo 데이터셋 준비
- `scripts/export_final_answers.py`
  review JSONL에서 최종 답안만 검증 후 추출
- `scripts/smoke_test.py`
  end-to-end 연결 확인

## 현재 스키마 규칙

- 좌표는 모두 normalized `0~1`
- `best_crop`은 `x1 < x2`, `y1 < y2`
- `reason`은 영어 한 문장
- `reason` 최대 40단어
- markdown, speculative wording 금지

## 자주 막히는 문제

### `image path does not exist`

- `image`가 상대경로인데 `--root-dir`를 빠뜨린 경우가 가장 흔합니다.

### teacher 출력이 reject되는 경우

- JSON object가 아님
- `best_crop` 좌표 범위가 잘못됨
- `reason`이 비어 있음
- `reason`이 여러 문장임

### GRPO가 바로 죽는 경우

- `outputs/sft_merged`가 없음
- `grpo_train.jsonl`이 비어 있음
- `reference_bbox`, `reference_reason`가 없음
- 학습 샘플 수가 너무 많음

처음에는 `--rl-max-samples 100` 정도로 작게 시작하는 편이 안전합니다.
