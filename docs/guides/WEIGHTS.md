# 모델 가중치 다운로드와 설치

추론 전에 한 번 가중치를 준비합니다. 명령은 **`foldforge weights download`**이며,
여러 모델을 한 번에 지정할 수 있습니다. 원본 다운로드 → FoldForge 형식으로 변환 →
SHA-256 검증까지 처리합니다. `ff.load_model()`은 다운로드를 자동으로 시작하지 않습니다.

## 빠른 시작

Pixi 체크아웃의 기본 환경에는 `weights`와 `language` extra가 포함됩니다.
`pixi run foldforge weights download --all-public`로 실행할 수 있습니다.
아래 `pip` 설치 단계는 별도 Python 환경에서 사용하는 경우입니다.

다운로드·변환 의존성을 설치합니다. 이미 설치한 FoldForge와 같은 환경에서 실행하세요.
소스 체크아웃에서는 `pip install -e '.[weights]'`를 사용합니다.

```bash
pip install 'foldforge[weights]'

# 공유 스토리지를 쓰려면 다운로드와 추론 모두 같은 환경 변수를 설정합니다.
export FOLDFORGE_CHECKPOINT_DIR=/data/foldforge/checkpoints

# 여러 모델을 한 번에 준비
foldforge weights download boltz2 opendde

# 지원 모델과 자동 설치 가능 여부
foldforge weights list

# 다운로드 없이 전체 작업 계획과 원본 파일 용량 확인
foldforge weights download --all-public --dry-run

# 자동 설치 가능한 모델 전체
foldforge weights download --all-public

# 로컬 예외까지 포함해 전체 모델 시도
foldforge weights download --all
```

`--all`은 12개 모델을 모두 대상으로 합니다. **현재 외부 원본에서 자동 설치하는 모델은
11개**입니다. Protenix2는 사용자 지정 로컬 가중치를 사용하는 예외입니다.
실패가 하나라도 있으면 마지막에 모델별 결과를 출력하고 종료 코드 `1`을 반환합니다.
모든 모델이 설치됐다는 의미로 `--all`을 해석하면 안 됩니다.

기본 저장 위치는 `$FOLDFORGE_CHECKPOINT_DIR`, 설정하지 않았다면
`$FOLDFORGE_HOME/checkpoints` (`FOLDFORGE_HOME` 기본값: `~/.cache/foldforge`)입니다.
한 번만 위치를 지정하려면 `--root /data/foldforge/checkpoints`를 사용하세요.
이 경우 추론 프로세스에도 같은 `FOLDFORGE_CHECKPOINT_DIR`을 설정해야 합니다.

```python
import foldforge as ff

model = ff.load_model("boltz2")
inputs = ff.load_inputs("input.yaml")
result = model(inputs)
```

## 모델별 설치 경로

모델명은 Python API와 CLI가 공유합니다. 아래 GB는 변환된 주 가중치의 십진 용량이며,
원본 캐시와 별도 언어모델 용량은 제외합니다.

| 모델명 | 주 가중치 경로 | GB | 외부 원본에서 자동 설치 | 원본 / 추가 조건 |
|---|---|---:|---|---|
| `boltz2` | `boltz2/boltz2.bin.zst` | 1.88 | 가능 | [Boltz-2](https://huggingface.co/boltz-community/boltz-2)의 confidence 가중치. Affinity 모델은 포함하지 않음 |
| `opendde` | `opendde/opendde.bin.zst` | 2.47 | 가능 | [OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE)의 `opendde.pt` |
| `openfold3` | `openfold3/openbind0.bin.zst` | 1.37 | 가능 | [OpenFold3](https://github.com/aqlaboratory/openfold-3)의 OpenBind 0 릴리스 |
| `openfold3-preview2` | `openfold3/openfold3.bin.zst` | 1.37 | 가능 | OpenFold3 preview-2. 위 모델과 다른 가중치 |
| `rosettafold3` | `rosettafold3/rosettafold3.bin.zst` | 1.36 | 가능 | [Foundry RF3](https://github.com/RosettaCommons/foundry/blob/production/models/rf3/README.md)의 `rf3_foundry_01_24_latest_remapped.ckpt` |
| `esmfold2` | `esmfold2/esmfold2.bin.zst` | 0.87 | 가능 | [ESMFold2](https://huggingface.co/biohub/ESMFold2)의 고정된 이전 릴리스 + LM sidecar + ESM-C 6B |
| `esmfold2-fast` | `esmfold2-fast/esmfold2_fast.bin.zst` | 0.70 | 가능 | [ESMFold2-Fast](https://huggingface.co/biohub/ESMFold2-Fast)의 고정된 이전 릴리스 + LM sidecar + ESM-C 6B |
| `af3` | `af3/af3.bin.zst` | 1.02 | 가능 | [Google 공식 직접 배포](https://github.com/google-deepmind/alphafold3#obtaining-model-parameters)의 고정된 객체 버전. 별도 변환 불필요 |
| `chai1` | `chai1/chai1.bin.zst` | 1.20 | 가능 | 원본 6개 파일 + ESM2 3B. 고정된 CPU 변환 프로필로 검증 |
| `intellifold2` | `intellifold2/intellifold2.bin.zst` | 3.16 | 가능 | 공통 컨테이너로 기록. 파라미터 값은 기존 원본 변환과 비트 단위 일치 |
| `protenix1` | `protenix/protenix1.bin.zst` | 0.99 | 가능 | v1.0.0 릴리스. FoldForge 규격으로 검사하며 JAX 불필요 |
| `protenix2` | `protenix/protenix2.bin.zst` | 1.33 | 로컬 가져오기 | 공개 다운로드 대상에서 제외. 사용할 권한이 있는 로컬 변환 파일을 `--from-dir`로 가져오기 |

변환 제약과 수동 변환 절차는 [CHECKPOINTS.md](CHECKPOINTS.md)에 기록되어 있습니다.
원본 가중치를 내려받을 수 있다는 사실과 현재 FoldForge 체크포인트로 검증됐다는 사실은
구분합니다. 이 명령은 잠금 파일을 바꾸거나 불일치한 파일을 설치하지 않습니다.

## 언어모델도 함께 설치

ESMFold2 두 모델에는 각각의 `*.lm.npz`와 공통 `esmc-6b/` 디렉터리가 필요합니다.
ESMFold2 실행에는 `pip install 'foldforge[language]'`도 필요합니다.
공통 [ESM-C 6B](https://huggingface.co/biohub/ESMC-6B)의 원본은 약 **25.4 GB**이며,
두 모델을 함께 요청하면 한 번만 내려받습니다. 필요한 config, shard index, 모든 가중치
shard와 tokenizer 파일을 고정된 리비전에서 받습니다.

ESMFold2는 `8fc3ff471022fdce52c77030685eb775de0c00a3`, Fast는
`c6c7958d63f5f2f1f0fed0bb9462316f8ccceea6`, ESM-C는
`45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a` 리비전을 사용합니다.
최신 HEAD는 이 변환기/언어모델 로더와 호환되지 않으므로 임의로 대체하지 마세요.

Chai-1을 로컬에서 가져올 때는 다음 파일도 원본 디렉터리에 있어야 합니다.

```text
chai1/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt
```

## 이미 검증된 가중치를 가져오기 / 오프라인 설치

본인이 사용할 권한이 있는 체크포인트 디렉터리가 있다면 네트워크 없이 가져옵니다.
원본 디렉터리는 위 표와 같은 구조여야 하며, 원본 파일은 변경하지 않습니다.
공유 언어모델과 sidecar도 함께 복사하고 SHA-256을 확인합니다.

```bash
foldforge weights download boltz2 chai1 --from-dir /shared/authorized-checkpoints
foldforge weights download --all --from-dir /shared/authorized-checkpoints
```

AF3도 공식 배포에서 직접 받은 파일을 `af3/af3.bin.zst` 위치에 준비한 뒤 가져옵니다.
AF3 공식 직접 다운로드는 [공식 안내](https://github.com/google-deepmind/alphafold3#obtaining-model-parameters)를 참고하세요.
AF3 파라미터의 사용과 전달은 [공식 이용 조건](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)을 따릅니다.
자동 명령이 접근 권한을 부여하거나 이용 조건에 대신 동의하지 않습니다.
직접 받은 릴리스가 승인된 두 잠금 파일의 어느 버전과도 일치하지 않으면 설치에 실패합니다.

## 검증, 재실행, 용량

- 원본과 설치 파일을 **크기와 전체 SHA-256**으로 확인합니다. 변환 결과는 배포된
  `checkpoints.lock` 또는 검증된 추가 릴리스인 `conversions.lock`과 일치해야 합니다.
  LM sidecar와 언어모델에도 별도 해시를 적용합니다.
- 설치 파일이 이미 정상이라면 재사용합니다. 알 수 없는 기존 파일을 덮어쓰지 않습니다.
  기존 파일의 검증이 실패하면 해당 파일을 별도로 보관한 뒤 다시 실행하세요.
- Hugging Face 전송은 해당 라이브러리의 캐시·재개 기능을 사용합니다. 공개 HTTPS 직링크는
  중단된 파일만 처음부터 다시 받습니다. 재실행해도 완료한 파일은 다시 받지 않습니다.
- 원본 캐시와 변환 후보는 `<root>/.weights/`에 보관합니다. 실패한 변환은 추론 경로에
  설치하지 않으며, 로그는 `.weights/converted/<model>/conversion.log`에 남습니다.
- 원본 캐시·작업용 복사본·변환 결과·최종 설치본이 공간을 사용합니다. `--dry-run` 용량은
  원본 파일 합계이며 설치에 필요한 전체 디스크 공간이 아닙니다. 모든 자동 모델을 처음
  설치한다면 충분한 여유 공간(예: 120 GB 이상)을 준비하세요.
- 변환은 GPU 없이 CPU에서 순차 실행합니다. Git과 변환 메모리가 필요합니다.
  서버에서는 로그인 노드 대신 작업 노드에서 실행하세요.

```bash
# 모든 주 가중치의 해시 검증. 설치하지 않은 모델도 missing으로 표시됩니다.
foldforge checkpoints verify

# 배포에 고정된 원본 URL, 리비전, 해시 확인
foldforge weights list --json
```

Hugging Face 접근 오류가 나면 해당 모델 페이지의 접근 조건과 로그인 상태를 확인하세요.
필요한 경우 `HF_TOKEN` 또는 Hugging Face의 로그인 기능을 사용합니다. 토큰은 명령 인자로
받지 않으며 일반 HTTPS 다운로드 서버로 보내지 않습니다.

## JAX 없는 변환 프로필과 호환성

추론은 PyTorch, CPU 입력 전처리는 NumPy와 표준 `dataclasses`를 사용합니다.
변환기에도 JAX·jaxlib·Haiku·Chex가 필요하지 않습니다. Protenix와 ESMFold2의
설정 검사는 FoldForge의 모델 규격을 사용합니다. 코드의 `haiku.py`는 AF3 바이너리
레코드 형식을 읽고 쓰는 로컬 코덱이며 외부 Haiku 프레임워크를 사용하지 않습니다.

CPU 변환 프로필 `cpu-v1`은 NumPy 2.5.3, zstandard 0.25.0과 단일 스레드 OpenBLAS
Haswell 경로를 고정합니다. 변환은 x86-64/AVX2 CPU에서 별도 프로세스로 실행됩니다.
이 설정은 추론 프로세스의 연산 설정을 바꾸지 않습니다.

기존 `checkpoints.lock`의 해시는 유지합니다. IntelliFold2, Chai1, Protenix1, Protenix2의 추가
변환 릴리스는 `conversions.lock`에 정확한 해시를 따로 등록했습니다. 새로운 크기의
파일은 해시까지 확인해야 로드할 수 있으며, 기존에 검증된 파일도 계속 사용할 수 있습니다.
승인되지 않은 해시를 가진 파일은 설치하지 않습니다.
AF3 공식 직접 배포본도 정확한 해시를 추가 등록했고, 엄격한 파라미터 로딩과 JAX 없는
CPU 및 GPU 추론을 확인했습니다. AF3는 변환 없이 원본 바이트 그대로 설치합니다.

네 모델은 각각 단백질(5I28), 리간드 복합체(3PTB), 다중 사슬 복합체(1A1K)에 대해
3개 seed × 5개 sample로 기존 릴리스 참조 판정 기준을 통과했습니다. 파라미터 비교에서
IntelliFold2의 수치는 비트 단위로 같고, Chai1과 Protenix1의 차이는 FP32 합성 연산의
반올림에 국한됩니다. Protenix2는 로컬 원본으로 변환해 참조 판정을 통과했습니다.
기존 수치 판정 기준을 완화하지 않았습니다.
