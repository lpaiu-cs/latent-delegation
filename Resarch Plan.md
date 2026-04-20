# 연구계획서

## 1. 제목

**Master-State-Preserving One-Way Latent Delegation Between Same-Family Open Models**

## 2. 문제 정의

본 연구는 같은 계열의 open-weight 모델 쌍에서, **큰 모델이 master state를 유지한 채 작은 모델에게 중간 보조 계산을 latent-space로 위임**할 수 있는지를 검증한다.

핵심은 다음입니다.

* 큰 모델이 최종 상태와 logits의 소유권을 가진다.
* 작은 모델은 토큰을 생성하지 않는다.
* 작은 모델은 큰 모델의 중간 hidden state를 projection으로 받아, 일부 중간 계산을 수행한 뒤, **전체 상태 복원이 아니라 low-rank residual correction**만 큰 모델에 돌려준다.
* 따라서 이 연구는 “작은 모델이 큰 모델을 대체하는가”가 아니라, **“작은 모델이 제거된 큰 모델의 중간 블록 일부를 보완할 수 있는가”**를 묻는다.

이 문제 정의는 LRT의 same-family affine alignment 근거를 활용하면서도, *Neural Incompatibility*가 경고한 “완전한 cross-scale parametric transfer”라는 과장을 피합니다. ([arXiv][1])

## 3. 연구 가설

가설은 세 문장으로 요약할 수 있습니다.

1. 같은 family의 대응 층에서는 큰 모델의 residual state를 작은 모델의 residual space로 보내는 **압축 affine projection**이 학습 가능하다. ([arXiv][1])

2. 작은 모델의 frozen suffix는 이 projected state를 받아, 제거된 큰 모델 중간 블록을 대체하는 데 유용한 **보조 latent computation**을 만들 수 있다.

3. 큰 모델이 최종 master state를 유지하고, 작은 모델이 full state가 아니라 **low-rank delta만 반환**하도록 설계하면, large→small 방향의 비가역성 문제를 피하면서도 유의미한 성능 회복이 가능하다.

## 4. 제안 방법

### 4-1. 전체 구조

큰 모델 9B를 세 부분으로 나눕니다.

* **Large prefix**: 9B의 앞부분
* **Delegated small suffix**: 2B의 중간~후반 일부 층
* **Large suffix**: 9B의 뒷부분

구조는 다음과 같습니다.

[
h^L_{k,t} = L_{1:k}(x_{\le t})
]

[
\hat h^S_t = \mathrm{RMSNorm}(A h^L_{k,t} + b)
]

[
s_t = S_{j:j+r'-1}(\hat h^S_{\le t})
]

[
\Delta_t = U(V^\top s_t)
]

[
\tilde h^L_{k+r,t} = h^L_{k,t} + \Delta_t
]

[
z_t = L_{k+r+1:n}(\tilde h^L_{k+r,\le t})
]

여기서 중요한 점은 **large model의 original middle block을 exact하게 재현하려는 것이 아니라**, 그 블록을 **small delegate block + return adapter**로 대체한 새로운 hybrid network를 학습한다는 것입니다. 이 점을 명확히 해야 연구가 과장되지 않습니다.

### 4-2. 권장 split

LRT는 대응 층 선택에서 **relative depth matching**을 강조합니다. Gemma 2 9B는 42층, 2B는 26층이므로, 9B의 24층은 상대 깊이 약 57%이고 2B의 대응 지점은 약 15층입니다. 그래서 초기 split은 다음처럼 잡는 것이 가장 자연스럽습니다. 

**1차 실험: conservative split**

* 9B prefix: layers 1–24
* 2B delegated suffix: layers 15–20
* 9B suffix: layers 31–42

즉, **9B의 25–30층 6개를 제거**하고, 그 자리를 **2B의 15–20층 6개**와 interface adapter가 메우는 구조입니다.

**2차 확장: moderate split**

* 9B prefix: layers 1–24
* 2B delegated suffix: layers 15–26
* 9B suffix: layers 37–42

즉, **9B의 25–36층 12개를 제거**하고, **2B의 15–26층 12개**로 대체합니다.

내 판단으로는 **주 실험은 1차 split**이어야 합니다. 목표가 SOTA가 아니라 feasibility라면, 처음부터 12개 large layer를 날리는 것은 과합니다. moderate split은 1차가 안정화된 후의 확장 실험으로 두는 것이 맞습니다.

### 4-3. 학습되는 파라미터

기본 설정에서는 backbone은 전부 frozen입니다.

* entry projection (A, b): (3584 \rightarrow 2304)
* return adapter (U, V): (2304 \rightarrow 3584)의 low-rank factorization
* optional: interface RMSNorm scale, 1개의 scalar gate

Gemma 2의 hidden size를 기준으로 하면, entry affine는 약 8.26M 파라미터이고, rank 64 return adapter는 약 0.38M 수준이라서, 총 학습 파라미터는 **9M 미만**으로 둘 수 있습니다. 이는 단일 GPU 한 달 실험에 맞는 수준입니다. 

## 5. 학습 절차

### Stage A. 표현 정렬

목표는 (h^L_k \rightarrow h^S_j) 정렬입니다.

* open English 텍스트 샘플과 reasoning prompt 샘플을 준비
* full 9B와 full 2B를 teacher-forced forward
* 같은 token position에서 9B의 (k)층 residual과 2B의 (j)층 residual을 수집
* (A,b)를 MSE + cosine loss로 먼저 학습

이 단계는 “큰 모델 상태를 작은 모델 suffix가 읽을 수 있는 작업공간으로 번역”하는 단계입니다.

### Stage B. 제거된 large block 복구

목표는 small suffix 출력으로 **제거된 large middle block의 효과**를 근사하는 것입니다.

* teacher target: full 9B의 (k+r)층 hidden state
* student input: projected 9B (k)층 hidden state를 small suffix에 통과시킨 출력
* 학습 대상: (U,V)
* loss: hidden MSE + cosine similarity

즉, 작은 모델이 “정답 상태 전체”를 복원하는 것이 아니라, 큰 모델이 다음 suffix를 잘 이어갈 수 있도록 필요한 correction만 돌려주게 합니다.

### Stage C. 최종 출력 정렬

마지막으로 hybrid model의 logits를 full 9B teacher에 맞춥니다.

* loss: KL(student logits, full 9B logits) + CE(next token)
* regularization: (|\Delta|) penalty
* backbone은 계속 frozen
* 학습 대상은 (A,b,U,V)와 optional gate만 유지

이 3단계 분리는 단일 GPU 예산에서 가장 현실적입니다. 처음부터 end-to-end로 몰아붙이면 실패 원인 분해가 어렵습니다.

## 6. 구현 제약과 자원 계획

RTX 5090은 공식적으로 32GB GDDR7 메모리를 갖습니다. bitsandbytes 문서는 4-bit quantization과 QLoRA 스타일의 “frozen base + extra trainable parameters” 흐름을 지원하고, 8/4-bit training은 extra parameters 학습에 맞춰져 있다고 설명합니다. LoRA는 pretrained weights를 freeze한 채 저랭크 파라미터만 학습하는 방식입니다. 따라서 이번 연구의 구조는 하드웨어 제약과 잘 맞습니다. ([NVIDIA][7])

실무 설정은 이렇게 두는 것이 안전합니다.

* backbone: 4-bit frozen loading
* interface/adapters: bf16
* sequence length: 256부터 시작, 안정화 후 512
* batch size: 1 또는 2
* gradient accumulation: 16~64
* optimizer: AdamW
* first target: 1차 split만 안정화

여기서 중요한 것은 **처음부터 LoRA를 small model에 얹지 않는 것**입니다. 먼저 “frozen small suffix + learned interfaces”만으로 되는지를 보는 것이 연구적으로 더 깨끗합니다. LoRA는 fallback이어야 합니다.

## 7. 비교 기준과 ablation

비교 기준은 최소한 다음 네 개가 있어야 합니다.

1. **Full 9B**
   상한선.

2. **Skip-only 9B**
   target middle block을 제거하고 아무 delegation 없이 바로 suffix로 연결.

3. **Bridge-only 9B**
   small model 없이, (h_k^L)에서 바로 low-rank bridge만 학습해 suffix에 연결.

4. **Proposed hybrid**
   9B prefix → 2B delegated suffix → 9B suffix.

이렇게 해야 “성능이 오른 이유가 단순한 bridge 때문인지, 실제로 small delegate computation이 도움이 된 것인지”를 구분할 수 있습니다.

Ablation은 다음이면 충분합니다.

* conservative split vs moderate split
* rank 16 / 32 / 64 / 128
* Stage A pre-alignment 유무
* optional gate 유무
* fallback으로 only-last-2-small-layers LoRA 유무

## 8. 평가 설계

1차 평가는 영어 벤치마크로 제한하는 것이 맞습니다. Gemma 2가 English-only이기 때문입니다. ([Google AI for Developers][8])

추천 평가는 다음입니다.

* **Held-out text perplexity / logit KL**
  general language degradation 확인용

* **GSM8K subset**
  grade-school math word problems 기반의 multi-step math reasoning 확인용 ([arXiv][9])

* **StrategyQA subset**
  implicit multi-hop boolean QA 확인용 ([arXiv][10])

* **MuSR small subset (optional)**
  natural narrative 기반 multistep soft reasoning 확인용. 다만 context가 길어 1차 실험에서는 optional로 두는 것이 맞습니다. ([arXiv][11])

평가 지표는 절대 점수만 보면 안 됩니다. 다음 네 가지를 같이 봐야 합니다.

* hidden-state cosine / MSE
* logit KL / perplexity
* benchmark accuracy
* **recovery rate**
  [
  \frac{\text{Acc}*{\text{proposed}}-\text{Acc}*{\text{skip}}}{\text{Acc}*{\text{full9B}}-\text{Acc}*{\text{skip}}}
  ]

이 recovery rate가 이번 연구의 핵심 지표입니다. 절대 성능이 full 9B보다 낮아도, skip-only 대비 성능을 유의미하게 회복하면 PoC로는 충분합니다.

## 9. 성공 기준

한 달짜리 feasibility 연구라면 성공 기준은 이 정도가 적절합니다.

* hidden reconstruction이 bridge-only baseline보다 명확히 좋을 것
* GSM8K/StrategyQA 중 최소 2개에서, **skip-only가 잃어버린 성능의 30% 이상**을 회복할 것
* full 9B 대비 **실제 wall-clock에서 10% 내외 이상의 속도 이득**이 관찰될 것
* peak VRAM이 5090 한 장에서 관리 가능할 것
* moderate split이 실패하더라도 conservative split에서 유의미한 recovery를 보일 것

즉, 이 연구의 성공은 “작은 모델이 큰 모델을 이겼다”가 아니라, **“같은 family에서 one-way latent delegation이 의미 있는 계산 단위로 작동한다”**를 보이는 것입니다.

## 10. 일정

### 1주차

* Gemma 접근 환경 구성
* hidden dump / hook 파이프라인 구축
* Stage A alignment
* conservative split만 우선 검증

### 2주차

* Stage B hidden recovery
* bridge-only baseline 완성
* hidden cosine / MSE 비교

### 3주차

* Stage C logit distillation
* GSM8K / StrategyQA subset 평가
* latency / VRAM profiling

### 4주차

* moderate split 확장 여부 판단
* optional MuSR subset
* ablation
* 실패 원인 분석 및 write-up

## 11. 리스크와 fallback

가장 큰 리스크는 이것입니다. **기존 문헌이 강하게 보여 준 것은 small→large affine transfer이지, large→small compressive delegation이 아닙니다.** 즉, 이번 연구의 핵심 novelty가 동시에 핵심 실패 지점입니다. 여기에 *Neural Incompatibility*가 지적한 cross-scale transfer의 어려움까지 겹칩니다. 또 LRT, model stitching, LLMBoost의 일부 핵심 근거는 preprint 중심이므로, 과도한 일반화는 금물입니다. ([arXiv][1])

그래서 fallback은 미리 정해 두는 것이 맞습니다.

* **Fallback A**: moderate split 폐기, conservative split만 유지
* **Fallback B**: small suffix 전체가 아니라 마지막 2개 층만 사용
* **Fallback C**: frozen interface만으로 안 되면 small suffix의 마지막 2개 층에만 rank-8 LoRA 추가
* **Fallback D**: end-task improvement가 약하면 논문 claim을 축소해
  “same-family large→small projection can recover hidden states/logits better than trivial bridges under a single-GPU budget”로 마감

이 fallback은 패배가 아닙니다. 오히려 이번 분야는 negative result도 정보가 큽니다.

## 12. 한 줄 결론

이 제약하에서 가장 타당한 연구는 **Gemma 2 9B/2B 기반의 same-family hybrid model**을 만들고, **9B의 중간 블록 일부를 2B suffix와 low-rank return adapter로 대체**해, **큰 모델이 master state를 유지한 채 작은 모델에게 latent 보조 계산을 일방향 위임할 수 있는지**를 검증하는 것입니다. SOTA를 노릴 계획은 아니고, **skip-only 대비 의미 있는 recovery + 약한 속도 이득**만 보여도 연구로서 충분히 가치가 있습니다. 

다음 단계는 이 계획서를 바로 실행 가능한 수준으로 내리는 것입니다: 실험 체크리스트, PyTorch 모듈 구조, hook 위치, loss 정의, 그리고 1주차용 minimal training loop.

[1]: https://arxiv.org/html/2506.00653v1 "Linear Representation Transferability Hypothesis: Leveraging Small Models to Steer Large Models"
[2]: https://arxiv.org/html/2506.06609v1 "Transferring Features Across Language Models With Model Stitching"
[3]: https://aclanthology.org/2025.acl-long.1047/ "Neural Incompatibility: The Unbridgeable Gap of Cross-Scale Parametric Knowledge Transfer in Large Language Models - ACL Anthology"
[4]: https://openreview.net/forum?id=tG4SgayTtk "Training Large Language Model to Reason in a Continuous Latent Space | OpenReview"
[5]: https://arxiv.org/pdf/2512.22309 "LLMBoost: Make Large Language Models Stronger with Boosting"
[6]: https://arxiv.org/abs/2106.09685 "[2106.09685] LoRA: Low-Rank Adaptation of Large Language Models"
[7]: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/ "GeForce RTX 5090 Graphics Cards | NVIDIA"
[8]: https://ai.google.dev/gemma/docs/core/model_card_2 "Gemma 2 model card  |  Google AI for Developers"
[9]: https://arxiv.org/abs/2110.14168?utm_source=chatgpt.com "[2110.14168] Training Verifiers to Solve Math Word Problems"
[10]: https://arxiv.org/abs/2101.02235?utm_source=chatgpt.com "Did Aristotle Use a Laptop? A Question Answering ..."
[11]: https://arxiv.org/abs/2310.16049?utm_source=chatgpt.com "MuSR: Testing the Limits of Chain-of-thought with Multistep ..."
