# Qwen3.5-0.8B 이미지 캡셔닝 파이프라인

`Qwen/Qwen3.5-0.8B`를 `ms-swift`로 `LoRA SFT -> merge -> LoRA GRPO -> merge` 순서로 학습하는 최소 파이프라인입니다.

이 저장소가 하는 일은 4가지입니다.

- 원본 JSONL을 검증하고 `ms-swift` 학습용 JSONL로 변환
- SFT / GRPO 실행 명령을 고정된 설정으로 래핑
- LoRA adapter merge, 추론, 평가까지 연결
- GPU 없이도 로컬에서 스모크 테스트 가능

학습 자체는 `Linux + NVIDIA GPU + CUDA` 기준입니다. `macOS`는 데이터 준비, 문서 수정, 테스트, 스모크 테스트 용도로 보는 것이 맞습니다.

## 빠른 시작

### 1. 로컬에서 바로 점검하기

GPU 없이도 아래 순서로 저장소가 끝까지 연결되는지 확인할 수 있습니다.

```bash
conda create -n qwen-lora-dev python=3.10 -y
conda activate qwen-lora-dev
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .[dev]
python scripts/check_environment.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/smoke_test.py
```

`smoke_test.py`는 실제 학습을 돌리지 않습니다. 대신 아래를 확인합니다.

1. 데모 데이터셋 생성
2. `prepare_dataset.py` 실행
3. `run_sft.py --dry-run`
4. `run_grpo.py --dry-run`
5. 더미 예측 평가

가장 먼저 확인할 엔트리포인트는 이 3개입니다.

```bash
python scripts/check_environment.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/smoke_test.py
```

### 2. GPU 서버에서 학습 환경 만들기

권장 기준:

- OS: Ubuntu 22.04 계열
- GPU: RTX 4090 24GB 1장 이상
- Python: 3.10
- 저장공간: 100GB 이상

```bash
conda env create -f environment.yml
conda activate qwen-lora
python -m pip install --upgrade pip setuptools wheel
```

그다음 서버 CUDA 버전에 맞는 `torch`를 먼저 설치합니다. 예시는 CUDA 12.1 기준입니다.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

이후 학습 의존성을 설치합니다.

```bash
python -m pip install -e .[train,dev]
python scripts/check_environment.py --expect-train
```

주의:

- `.[train]`에는 `vllm`이 포함되어 있어 로컬 macOS에서는 보통 실패합니다.
- 학습은 GPU 서버에서만 진행하는 편이 안전합니다.

## 실제 파이프라인

### 1. 원본 데이터 준비

입력은 JSONL입니다. 한 줄에 샘플 하나를 넣습니다.

```json
{"image":"images/dog_001.jpg","description":"A dog running on grass.","caption":"A brown dog runs across a grassy field.","split":"train","quality_score":0.98,"sample_id":"dog-001"}
```

필수 필드:

- `image` 또는 `image_path`
- `description`
- `caption`

선택 필드:

- `split`: `train`, `val`, `test`
- `quality_score`: RL subset 우선순위
- `sample_id`: 샘플 식별자

캡션 규약:

- 영어 한 문장
- 최대 40단어
- 마크다운, 리스트, 제목 금지
- `maybe`, `perhaps`, `probably` 같은 추측 표현 금지

### 2. 데이터셋 변환

가장 빠른 수동 검증은 저장소에 포함된 데모 데이터셋으로 해보는 것입니다.

```bash
python scripts/prepare_dataset.py \
  --input examples/demo_dataset/annotations.jsonl \
  --output-dir data/processed \
  --root-dir examples/demo_dataset
```

정상이라면 아래 파일이 생성됩니다.

- `data/processed/sft_train.jsonl`
- `data/processed/sft_val.jsonl`
- `data/processed/grpo_train.jsonl`
- `data/processed/audit.json`

`audit.json`에서 유효 행 수와 invalid row를 먼저 확인하면 됩니다.

추가 옵션:

```bash
python scripts/prepare_dataset.py \
  --input /data/my_dataset/annotations.jsonl \
  --output-dir data/processed \
  --root-dir /data/my_dataset \
  --skip-invalid \
  --val-ratio 0.05 \
  --rl-max-samples 1000
```

### 3. 학습 명령 먼저 확인

실제 실행 전에 명령부터 보는 편이 안전합니다.

```bash
python scripts/run_sft.py --dry-run
python scripts/run_grpo.py --dry-run
```

기본 설정은 각각 [configs/sft.yaml](/Users/rhino/Documents/gitRepo/qwen_lora/configs/sft.yaml) 과 [configs/grpo.yaml](/Users/rhino/Documents/gitRepo/qwen_lora/configs/grpo.yaml) 에 있습니다.

데이터 경로나 출력 경로를 바꾸려면 설정 파일을 복사해서 수정한 뒤 `--config`로 넘기면 됩니다.

```bash
python scripts/run_sft.py --config configs/sft.yaml
python scripts/run_grpo.py --config configs/grpo.yaml
```

### 4. SFT

```bash
python scripts/run_sft.py
```

기본 출력 위치는 `outputs/sft`입니다.

### 5. SFT adapter merge

```bash
python scripts/merge_adapter.py \
  --adapter-root outputs/sft \
  --output-dir outputs/sft_merged
```

`checkpoint-*`가 여러 개면 가장 최근 체크포인트를 자동으로 찾습니다.

### 6. GRPO

```bash
python scripts/run_grpo.py
```

기본 설정에서는 `outputs/sft_merged`를 시작 모델로 사용하고 출력은 `outputs/grpo`에 저장합니다.

### 7. GRPO adapter merge

```bash
python scripts/merge_adapter.py \
  --adapter-root outputs/grpo \
  --output-dir outputs/final_merged
```

### 8. 추론

단일 이미지:

```bash
python scripts/infer.py \
  --model outputs/final_merged \
  --image /data/my_dataset/images/dog_001.jpg \
  --description "A dog running on grass."
```

배치 추론:

```bash
python scripts/predict_dataset.py \
  --model outputs/final_merged \
  --input data/processed/sft_val.jsonl \
  --output outputs/predictions/final_val_predictions.jsonl
```

### 9. 평가

단일 결과 평가:

```bash
python scripts/evaluate.py \
  --predictions outputs/predictions/final_val_predictions.jsonl
```

SFT와 RL 결과 비교:

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

Acceptance gate 기준:

- `bert_score_f1`가 `+0.01` 이상 개선되거나
- `rouge_l`이 `+2.0` 이상 개선
- `format_pass_rate >= 98`
- `empty_output_rate == 0`

이 기준을 넘지 못하면 `outputs/final_merged` 대신 `outputs/sft_merged`를 유지하는 편이 낫습니다.

## 기본 파일 구조

- [scripts/prepare_dataset.py](/Users/rhino/Documents/gitRepo/qwen_lora/scripts/prepare_dataset.py): 원본 JSONL 검증 및 SFT/GRPO JSONL 생성
- [scripts/run_sft.py](/Users/rhino/Documents/gitRepo/qwen_lora/scripts/run_sft.py): SFT 실행 래퍼
- [scripts/run_grpo.py](/Users/rhino/Documents/gitRepo/qwen_lora/scripts/run_grpo.py): GRPO 실행 래퍼
- [scripts/merge_adapter.py](/Users/rhino/Documents/gitRepo/qwen_lora/scripts/merge_adapter.py): 최신 adapter merge
- [scripts/infer.py](/Users/rhino/Documents/gitRepo/qwen_lora/scripts/infer.py): 단일 이미지 추론
- [scripts/predict_dataset.py](/Users/rhino/Documents/gitRepo/qwen_lora/scripts/predict_dataset.py): 배치 추론
- [scripts/evaluate.py](/Users/rhino/Documents/gitRepo/qwen_lora/scripts/evaluate.py): ROUGE-L / BERTScore / format 평가
- [scripts/smoke_test.py](/Users/rhino/Documents/gitRepo/qwen_lora/scripts/smoke_test.py): 로컬 스모크 테스트
- [plugins/caption_reward.py](/Users/rhino/Documents/gitRepo/qwen_lora/plugins/caption_reward.py): GRPO reward 등록

## 자주 막히는 지점

### `image path does not exist`

- 상대경로를 썼는데 `--root-dir`를 빠뜨린 경우가 가장 흔합니다.
- 먼저 `prepare_dataset.py`부터 통과시키는 편이 맞습니다.

### `vllm` 또는 CUDA 관련 오류

- `torch`를 CUDA 버전에 맞게 설치하지 않으면 연쇄적으로 깨집니다.
- `python scripts/check_environment.py --expect-train` 결과부터 확인하면 됩니다.

### GRPO가 바로 죽는 경우

- `outputs/sft_merged`가 없거나 잘못 만들어진 경우
- `grpo_train.jsonl`이 비었거나 `reference_caption`이 없는 경우
- RL subset이 너무 큰 경우

처음에는 `--rl-max-samples 100` 정도로 작게 시작하는 편이 안전합니다.

## 포함된 샘플

- `examples/demo_dataset/`: 바로 `prepare_dataset.py`에 넣어볼 수 있는 정적 샘플
- `scripts/create_demo_dataset.py`: 임시 데모 데이터셋 생성
- `examples/raw_annotations.example.jsonl`: 원본 JSONL 포맷 예시

## 참고

- 프롬프트 템플릿은 [src/qwen_lora/prompting.py](/Users/rhino/Documents/gitRepo/qwen_lora/src/qwen_lora/prompting.py) 에 고정되어 있습니다.
- 보상 함수는 semantic / rouge / format 3개를 사용합니다.
- 로컬 스모크 테스트에서는 실제 학습 대신 dry-run만 수행합니다.
