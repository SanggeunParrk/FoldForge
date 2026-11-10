# Protenix v1 파인튜닝 (학습 스크립트, phase 0/1)

`scripts/train_protenix1.py`와 `src/foldforge/train/`는 FoldForge 모델(protenix1)을 **MiniWorld의 학습 레시피**로 파인튜닝한다.
현재 구현 범위는 phase 0(기울기·엔진 backward 확인)과 phase 1(eager 스텝, 단일 GPU, gradient accumulation)이다. 데이터 로더(phase 2)와
스텝 전체 CUDA 그래프(phase 3)는 아직 없다.

## 레시피 (`foldforge.train.config.TrainConfig`)

출처는 MiniWorld의 `configs/miniworld/phase2a_diffusion_medium_v200_b200_joint.yaml` 계열, `loss/diffusion_v102_ft.yaml`, `train/default.yaml`이다
(Protenix의 `finetune_demo.sh`가 아니다).

| 항목 | 값 |
|---|---|
| 손실 | diffusion 4.0, alpha_dna 5, alpha_rna 5, alpha_ligand 10, bond 1.0, distogram 0.03, smooth LDDT 0.0(미구현) |
| 옵티마이저 | Adam(0.9, 0.95), max_lr 5e-3, min_lr 1e-4, warmup 1000, 5e4 스텝마다 x0.95 |
| clip / EMA | grad clip 1.0, EMA 0.999 |
| 유효 배치 | 배치 1 x grad_accum 32 x GPU 수 |
| 정밀도 | fp32 마스터 가중치 + bf16 autocast, 손실은 fp32 |
| recycle | 1~4 중 균등 추출, 마지막 pass만 기울기 |
| 확산 | 증강 48개, EDM P_mean -1.2, P_std 1.5, sigma_data 16 |

## 경고: 학습률

`max_lr=5e-3`은 MiniWorld에서 **동결된 트렁크 + 새로 초기화한 확산 헤드**에 쓰는 값이다. 사전학습된 Protenix 트렁크를 풀어 이 값으로 돌리면 가중치가
망가질 수 있다. 스모크(고정 예제 한 개, warmup 0)에서 5e-3은 처음 10스텝 평균 손실이 4.5배로 튀었다가(시작 1.02 -> 4.50) 50스텝 안에 회복했다.
1e-4는 안정적으로 감소했다. 실제 파인튜닝에서는 `--trunk-lr-scale`(트렁크 LR 배율)이나 `--max-lr`를 낮춰서 시작하는 것을 권한다.

## 죽은 모듈

Protenix v1에는 모든 파라미터가 1e-37 수준인 모듈이 있다(추론에서는 건너뛴다, [CONFIG.md](CONFIG.md)의 `engine.skip_dead_modules`). 학습에서는 이 모듈도 돌아야
파라미터가 학습된다. 학습 스크립트는 `config={"engine": {"skip_dead_modules": False}}`로 모델을 읽고, `prepare.enable_training`이 남은 `dead` 플래그도 지운다.
알려진 한계: 이 파라미터 494개는 bf16에서 기울기가 0으로 언더플로하고(죽은 가중치 자체가 1e-37), 엔진 쪽 해결은 `ENGINE-REQUESTS.md`에 없다
(R16은 atom-pair 조건화 기울기, R17은 `PoolAtoms` backward).

## 데이터

FoldForge는 MiniWorld를 import하지 않는다. 데이터 소스는 `--batch-source module:callable`이다. callable은 `(cfg, device)`를 받아 `TrainBatch`의
이터레이터를 돌려준다(`foldforge.train.data`). 어댑터는 MiniWorld 쪽에서 제공한다(BioMolDB 카탈로그와 LMDB 경로는 MiniWorld의
`configs/miniworld/data/H100_default.yaml`).

## 실행

```bash
# 스모크: 고정 예제 하나(FoldBench 타깃, 정답은 모델 자신의 예측으로 대체) 과적합 + 기울기 점검 + 커널 표 + 체크포인트 왕복
python scripts/train_protenix1.py --smoke --target 8ayg-assembly1 --steps 50 --warmup-steps 0 --run-dir RUN
# 학습
python scripts/train_protenix1.py --batch-source my_pkg.adapter:batches --steps 100000 --run-dir RUN [--ckpt X.miniworld.pt]
```

체크포인트는 네이티브 `*.miniworld.pt`(+ 같은 이름의 `.train.pt`: optimizer, EMA, step)로 저장된다. `protenix1.bin.zst` 형식으로의 역변환은 없다.
