# Qwen3.5-0.8B Detector-Aware Crop Recommendation Pipeline

이 저장소는 `이미지 + detector 결과 + auto crop top1 결과`를 입력으로 받아 최종 crop 추천 JSON을 생성하도록 student 모델을 학습하는 파이프라인입니다.

```json
{
  "best_crop": {
    "x1": 0.118,
    "y1": 0.06,
    "x2": 0.902,
    "y2": 0.948
  },
  "reason": "The crop keeps the main person fully visible, includes nearby salient objects, and removes empty background."
}
```

이 저장소가 학습하는 것은 detector나 auto-crop 모델이 아니라, 두 모델의 결과와 이미지를 보고 더 나은 최종 crop을 고르는 작은 student 모델입니다. 기본 흐름은 `teacher label 생성 -> SFT -> LoRA merge -> GRPO -> LoRA merge -> 평가/검수`입니다.

## 기본 경로

README와 기본 설정 파일은 아래 경로를 기준으로 맞춰져 있습니다.

- raw annotation: `data/raw/annotations.jsonl`
- teacher label 포함 raw annotation: `data/raw/annotations_with_teacher.jsonl`
- processed dataset: `data/processed/`
- SFT output: `outputs/sft`
- merged SFT model: `outputs/sft_merged`
- GRPO output: `outputs/grpo`
- merged GRPO model: `outputs/grpo_merged`

`configs/sft.yaml`과 `configs/grpo.yaml`도 이 경로를 사용합니다. 다른 경로를 쓰려면 README 명령과 config를 같이 바꾸세요.

## Raw Annotation Schema

한 줄이 하나의 이미지 샘플입니다. 좌표는 모두 normalized `[0, 1]`입니다.

```json
{
  "sample_id": "dog-001",
  "image": "images/dog_001.jpg",
  "split": "train",
  "detector_objects": [
    {
      "class_id": 16,
      "class_name": "dog",
      "box_cx": 0.599,
      "box_cy": 0.57,
      "box_w": 0.385,
      "box_h": 0.43
    }
  ],
  "autocrop_top1": {
    "x1": 0.34,
    "y1": 0.18,
    "x2": 0.9,
    "y2": 0.98,
    "score": 0.78
  },
  "teacher_answer": {
    "best_crop": {
      "x1": 0.3,
      "y1": 0.16,
      "x2": 0.92,
      "y2": 0.98
    },
    "reason": "The crop keeps the running dog fully visible while trimming empty sky and preserving enough grass for context."
  }
}
```

`teacher_answer`는 teacher label 생성 전에는 없어도 됩니다. `prepare_dataset.py`에는 반드시 포함되어 있어야 합니다.

## 로컬 검증

```bash
conda create -n qwen-lora-dev python=3.10 -y
conda activate qwen-lora-dev
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
python scripts/check_environment.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/smoke_test.py
```

이 단계는 실제 학습을 하지 않습니다. 데이터 검증, SFT/GRPO command 생성, 평가, 최종 답안 export까지의 배선을 확인합니다.

## 작업 폴더 초기화

```bash
python scripts/init_workspace.py --workspace-dir .
```

생성되는 주요 파일과 폴더입니다.

- `data/raw/annotations.template.jsonl`
- `data/demo_dataset/annotations.jsonl`
- `data/processed/`
- `outputs/predictions/`
- `outputs/reviews/`
- `outputs/final_answers/`

데모 데이터셋으로 데이터 준비가 되는지 먼저 확인하려면 아래를 실행합니다.

```bash
python scripts/prepare_dataset.py \
  --input data/demo_dataset/annotations.jsonl \
  --output-dir data/processed \
  --root-dir data/demo_dataset
```

## Teacher Label 생성

학습 전에 teacher 모델을 오프라인으로 돌려 `teacher_answer`를 채웁니다. 이미 `teacher_answer`가 있는 row는 기본적으로 재사용됩니다.

### ms-swift 로컬 teacher

```bash
python scripts/generate_teacher_labels.py \
  --model /path/to/teacher-model \
  --input data/raw/annotations.jsonl \
  --output data/raw/annotations_with_teacher.jsonl \
  --root-dir /data/my_dataset \
  --temperature 0.0
```

모든 row에 이미 `teacher_answer`가 있으면 `--model` 없이 검증과 정규화만 할 수 있습니다.

```bash
python scripts/generate_teacher_labels.py \
  --input data/raw/annotations_with_teacher.jsonl \
  --output data/raw/annotations_with_teacher.normalized.jsonl \
  --root-dir /data/my_dataset
```

### vLLM OpenAI-compatible server

vLLM 서버를 별도로 띄워 둔 경우 더 빠르게 teacher label을 만들 수 있습니다.

```bash
python -m pip install -e '.[teacher]'
python scripts/generate_teacher_labels_vllm_fast.py \
  --input data/raw/annotations.jsonl \
  --output data/raw/annotations_with_teacher.jsonl \
  --root-dir /data/my_dataset \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --served-model-name openbmb/MiniCPM-o-4_5 \
  --temperature 0.0 \
  --max-new-tokens 128 \
  --max-samples 100 \
  --max-workers 8 \
  --image-url-mode data_url
```

빠른 테스트처럼 일부만 생성하려면 `--max-samples 100`처럼 처리할 row 수를 제한하세요. 생략하면 전체 입력을 처리합니다.

기본값은 엄격 검증입니다. `autocrop_top1`이 없는 row를 full-image crop으로 처리해야 할 때만 `--fill-missing-autocrop`을 사용하고, 약간 벗어난 crop 좌표를 잘라서 진행해야 할 때만 `--clip-autocrop`을 사용하세요.

## COCO/Parquet 입력을 Raw JSONL로 변환

새로 추가된 `scripts/create_dataset_for_teacher.py`는 detector parquet와 COCO-style autocrop JSON을 teacher 생성용 raw JSONL로 합칩니다. import 시 자동 실행하지 않고 CLI로만 동작합니다.

```bash
python -m pip install -e '.[teacher]'
python scripts/create_dataset_for_teacher.py \
  --detector-parquet data/coco/train/detector_train2017.parquet \
  --autocrop-json data/coco/train/coco_annotations_train_onnx_gem.json \
  --output data/raw/coco_train_det_autocrop.jsonl \
  --split train \
  --image-prefix /data/coco/train2017
```

val 파일도 같은 방식으로 만들 수 있습니다.

```bash
python scripts/create_dataset_for_teacher.py \
  --detector-parquet data/coco/val/detector_val2017.parquet \
  --autocrop-json data/coco/val/coco_annotations_val_onnx_gem.json \
  --output data/raw/coco_val_det_autocrop.jsonl \
  --split val \
  --image-prefix /data/coco/val2017
```

`--missing-autocrop` 기본값은 `error`입니다. 누락을 의도적으로 제외하려면 `--missing-autocrop skip`, full-image로 대체하려면 `--missing-autocrop full_image`를 명시하세요.

## 학습용 JSONL 생성

teacher label이 채워진 raw JSONL을 SFT/GRPO 입력으로 변환합니다.

```bash
python scripts/prepare_dataset.py \
  --input data/raw/annotations_with_teacher.jsonl \
  --output-dir data/processed \
  --root-dir /data/my_dataset
```

train/val raw 파일이 따로 있다면 한 번에 넘기세요. 두 번 실행하면 같은 출력 파일을 덮어쓸 수 있습니다.

```bash
python scripts/prepare_dataset.py \
  --input data/raw/coco_train_with_teacher.jsonl data/raw/coco_val_with_teacher.jsonl \
  --output-dir data/processed
```

생성 파일입니다.

- `data/processed/sft_train.jsonl`
- `data/processed/sft_val.jsonl`
- `data/processed/grpo_train.jsonl`
- `data/processed/audit.json`

## GPU 학습 환경

```bash
conda env create -f environment.yml
conda activate qwen-lora
python -m pip install --upgrade pip setuptools wheel
# CUDA 버전에 맞는 PyTorch wheel index를 사용하세요.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[train,dev]'
python scripts/check_environment.py --expect-train
```

학습은 `Linux + NVIDIA GPU + CUDA` 기준입니다. macOS는 데이터 준비와 dry-run 검증 용도로 두는 편이 안전합니다.

## SFT

실행 전에 command와 경로를 확인합니다.

```bash
python scripts/run_sft.py --dry-run
python scripts/run_sft.py
```

기본 config는 아래 입력을 사용합니다.

- `data/processed/sft_train.jsonl`
- `data/processed/sft_val.jsonl`

## SFT Adapter Merge

```bash
python scripts/merge_adapter.py \
  --adapter-root outputs/sft \
  --output-dir outputs/sft_merged
```

## GRPO

```bash
python scripts/run_grpo.py --dry-run
python scripts/run_grpo.py
```

기본 config는 아래 입력을 사용합니다.

- model: `outputs/sft_merged`
- dataset: `data/processed/grpo_train.jsonl`

## GRPO Adapter Merge

```bash
python scripts/merge_adapter.py \
  --adapter-root outputs/grpo \
  --output-dir outputs/grpo_merged
```

## 추론

```bash
python scripts/infer.py \
  --model outputs/grpo_merged \
  --image /data/my_dataset/images/img-000001.jpg \
  --detector-objects-json detector_objects.json \
  --autocrop-top1-json autocrop_top1.json
```

## 데이터셋 예측과 검수 파일 생성

```bash
python scripts/predict_dataset.py \
  --model outputs/grpo_merged \
  --input data/processed/sft_val.jsonl \
  --output outputs/predictions/grpo_val_predictions.jsonl \
  --review-output outputs/reviews/grpo_val_review.jsonl
```

`--review-output`은 사람이 수정 가능한 JSONL입니다. `final_answer`를 수정한 뒤 최종 배포용 파일만 export합니다.

```bash
python scripts/export_final_answers.py \
  --input outputs/reviews/grpo_val_review.jsonl \
  --output outputs/final_answers/grpo_val_answers.jsonl
```

## 평가

SFT와 GRPO를 같은 validation set에서 예측한 뒤 비교합니다.

```bash
python scripts/predict_dataset.py \
  --model outputs/sft_merged \
  --input data/processed/sft_val.jsonl \
  --output outputs/predictions/sft_val_predictions.jsonl

python scripts/predict_dataset.py \
  --model outputs/grpo_merged \
  --input data/processed/sft_val.jsonl \
  --output outputs/predictions/grpo_val_predictions.jsonl

python scripts/evaluate.py \
  --predictions outputs/predictions/grpo_val_predictions.jsonl \
  --baseline-predictions outputs/predictions/sft_val_predictions.jsonl
```

`sentence-transformers`가 없는 로컬 환경에서 metric 배선만 확인할 때는 `--semantic-fallback`을 붙이면 lexical similarity로 대체합니다.

평가 항목입니다.

- JSON parse 성공률
- bbox 유효 비율
- teacher 대비 mean IoU
- teacher 대비 좌표 오차
- reason semantic similarity
- autocrop baseline 대비 개선 여부
- SFT 대비 GRPO 개선 여부

## 스키마 규칙

- 좌표는 모두 normalized `[0, 1]`
- `best_crop`은 `x1 < x2`, `y1 < y2`
- `reason`은 영어 한 문장
- `reason`은 최대 40단어
- markdown, list formatting, speculative wording은 reject

## 자주 보는 파일

- `src/qwen_lora/data_utils.py`: raw annotation 검증과 SFT/GRPO row 변환
- `src/qwen_lora/prompting.py`: 모델 입력 prompt
- `src/qwen_lora/reward_core.py`: JSON 파싱, bbox/reason 검증, 평가 metric
- `plugins/crop_reward.py`: GRPO reward 함수
- `configs/sft.yaml`: SFT 설정
- `configs/grpo.yaml`: GRPO 설정
- `scripts/create_dataset_for_teacher.py`: detector/autocrop 결과를 raw JSONL로 병합
- `scripts/generate_teacher_labels.py`: ms-swift 기반 teacher label 생성
- `scripts/generate_teacher_labels_vllm_fast.py`: vLLM 서버 기반 teacher label 생성
- `scripts/prepare_dataset.py`: 학습용 JSONL 생성
- `scripts/smoke_test.py`: 로컬 end-to-end smoke test
