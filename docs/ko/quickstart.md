# 한국어 빠른 시작

AIsketcher는 한 장의 스케치에서 여러 디자인 방향을 탐색하고, 선택한
결과의 변형과 재현 정보를 함께 보존하는 Python SDK입니다.

```text
준비 → 탐색 → 선택 → 변형 → 내보내기 → 재현
```

## 설치

AIsketcher 0.2.1은
[PyPI](https://pypi.org/project/AIsketcher/0.2.1/)에서 설치할 수 있습니다.
나중에 패키지가 갱신되어도 같은 환경을 만들 수 있도록 버전을 고정합니다.
PyPI 패키지명은 대소문자를 구분하지 않지만 Python import와 CLI에 맞춰 설치
대상은 소문자 `aisketcher`로 표기합니다.

최신 버전은 다음 한 줄로 설치합니다.

```bash
pip install aisketcher
```

동일한 환경을 재현하려면 아래처럼 공개 버전을 고정합니다.

기본 패키지에는 Torch, Diffusers, 모델 파일, Gradio 실행 환경이 포함되지
않습니다. Studio 코드와 가이드 샘플은 기본 wheel에 들어 있으며 `demo`
선택 의존성이 화면을 실행할 Gradio 환경을 추가합니다.

```bash
python -m pip install "aisketcher==0.2.1"
```

로컬 이미지 생성 기능은 선택 의존성으로 추가합니다.

```bash
python -m pip install "aisketcher[local]==0.2.1"
```

Studio 화면을 사용하려면 `demo` 선택 의존성을 설치합니다.

```bash
python -m pip install "aisketcher[demo]==0.2.1"
aisketcher init
aisketcher studio
```

한 줄로 처음 실행하려면 다음 명령을 사용합니다. 이미 설정 파일이 있다면
`aisketcher init &&` 부분은 빼세요.

```bash
python -m pip install "aisketcher[demo]==0.2.1" && aisketcher init && aisketcher studio
```

`init`은 최초 한 번 버전이 명시된 사용자 YAML 설정을 만들며 모델을 받지 않습니다.
기존 파일은 보호하고, 다른 위치가 필요할 때만 `--path`를 사용합니다.
Studio의 **Guided Sample**은 패키지에 포함된 익명 source, v0.2 Canny
control, 네 개의 실제 로컬 생성 후보, 선택 정보, 검증용 manifest를 읽기
전용으로 엽니다. 따라서 모델이나 네트워크 없이 바로 사용할 수 있습니다.
로컬 preset을 명시적으로 선택하고 설치 계획과 라이선스를 확인해야만 모델
다운로드가 시작됩니다.

실제 로컬 생성과 Studio를 함께 설치하려면 다음 명령을 사용합니다.

```bash
python -m pip install "aisketcher[local,demo]==0.2.1"
```

## 기본 흐름

```python
from aisketcher import FakeBackend, Intent, PresetManager, SeedPlan, Studio

preset = "sdxl-canny-lite@1"
models = PresetManager()
plan = models.plan_install(preset)
print(plan.license_notice, plan.estimated_bytes, plan.items)

# 저장소, revision, 예상 용량, 모델 라이선스를 확인한 뒤에만 실행합니다.
if not plan.installed:
    models.install(preset, confirm=True)

studio = Studio.from_preset(preset, device="auto", preset_manager=models)
prepared = studio.prepare("sketch.jpg")

study = studio.explore(
    prepared,
    intent=Intent(
        prompt="종이 공예 느낌의 친근한 판타지 왕국",
        profile="graphic_design",
        structure="balanced",
    ),
    outputs=4,
    seed_plan=SeedPlan.scout(4),
)

selected = study.pick(1)  # 0부터 세는 고정 인덱스이므로 두 번째 후보입니다.
variations = studio.vary(
    selected,
    outputs=4,
    strength="subtle",
    locks=("structure",),
)
variations.export("design-study")
```

모델을 받지 않고 API와 계보 저장만 시험하려면
`Studio(FakeBackend(), preset="sdxl-canny-lite@1")`를 사용할 수 있습니다.
이 결과는 코드 검증용 결정론적 fixture이며 생성 모델의 창작 결과가 아닙니다.

기본 출력은 네 장입니다. seed는 작품의 품질 점수가 아니라 한 번의
탐색을 다시 찾기 위한 좌표에 가깝습니다. AIsketcher는 실제 사용한 seed와
해석된 설정을 manifest에 기록합니다.

## Simple과 Advanced

- **Simple**: 스케치, 한 문장 설명, 작업 유형,
  `Loose / Balanced / Faithful`만 선택합니다.
- **Advanced**: 모델, Canny, seed, 출력 수, 변형 강도, lock, 재현 설정을
  조절합니다.

두 화면은 같은 작업 상태를 공유합니다. Advanced에서 값을 변경한 뒤
Simple로 돌아오면 `Advanced overrides active` 표시와 초기화 버튼이
나타납니다.

[![AIsketcher Studio 한국어 Simple 화면: 입력 스케치, 선택 결과, 네 개 방향과 다음 작업](../assets/aisketcher-studio-guided-sample-ko.jpg)](../assets/aisketcher-studio-guided-sample-ko.jpg)

*모델 없이 번들 가이드 샘플을 연 실제 로컬 화면입니다. 이미지를 누르면
크게 볼 수 있습니다. 왼쪽의 빈 입력란은 새 작업용이고 오른쪽에는 읽기
전용 fixture가 열려 있습니다. 모델 없이 비교·선택·내보내기를 체험할 수
있고, 새 탐색과 발전은 로컬 preset을 준비해 별도 study에서 진행합니다.*

## 알아둘 점

- 누구에게나 좋은 “추천 seed”는 없습니다. 구조 일치, edge 상태, 후보
  중복도 같은 기술적 관찰만 배지로 제공합니다.
- strict replay는 입력과 모델 revision이 달라지면 중단합니다. compatible
  replay는 대체된 항목을 보고하고 계속할 수 있습니다.
- 기본 내보내기는 이미지를 EXIF 없이 다시 저장하고, source 파일명을 빼며,
  backend metadata를 허용 목록으로 제한합니다. prompt, profile, custom backend
  식별자에는 token이나 비공개 경로를 넣지 마세요.
- 예제 드로잉과 결과 이미지는 MIT 라이선스 대상이 아닙니다.

다음으로 [전체 SDK 흐름](../sdk/workflow.md),
[seed와 출력 개수](../guides/seeds.md),
[환경설정 레퍼런스](../reference/configuration.md),
[문제 해결](../guides/troubleshooting.md),
[개인정보와 자산 처리](../guides/privacy.md)를 읽어보세요.
