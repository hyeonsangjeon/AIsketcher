"""Small, explicit translation catalog for the Studio example."""

from __future__ import annotations

from typing import Final

SUPPORTED_LANGUAGES: Final = ("en", "ko")

_TEXT: Final[dict[str, dict[str, str]]] = {
    "en": {
        "simple": "Simple",
        "advanced": "Advanced",
        "guide": "Guide",
        "headline": "Turn one sketch\ninto a design lineage.",
        "subhead": "Explore four directions, pick one, then refine it without losing the thread.",
        "sketch": "1  Sketch",
        "upload": "Upload sketch",
        "brief": "2  Creative brief",
        "brief_placeholder": "Describe the mood, material, palette, and intended use.",
        "profile": "Work type",
        "structure": "3  Structure",
        "loose": "Loose",
        "balanced": "Balanced",
        "faithful": "Faithful",
        "explore": "Explore directions",
        "guided": "Try the guided sample",
        "source": "Before (sketch)",
        "selected": "After (selected direction)",
        "directions": "Directions",
        "directions_help": "Different directions generated from your sketch and brief.",
        "empty": "Add a sketch, or open the bundled Guided Sample with no model download.",
        "refine": "Refine this direction",
        "again": "Try another direction",
        "export": "Finalize & export",
        "tip": "Tip: refine while keeping the structure locked.",
        "preset": "Preset",
        "seed_plan": "Seed plan",
        "auto_seeds": "Auto (different seeds)",
        "locked_seed": "Locked seed (same start)",
        "custom_seeds": "Custom seeds",
        "locked_seed_placeholder": "e.g., 6764547109648557242",
        "custom_seed_placeholder": "e.g., 12345, 67890, 13579, 24680",
        "outputs": "Outputs",
        "canny": "Canny control",
        "canny_info": "Required by the current SDXL presets",
        "steps": "Steps",
        "guidance": "Guidance",
        "variation": "Variation strength",
        "locks": "Variation locks",
        "lock_structure": "Lock structure",
        "manifest": "Manifest & replay",
        "manifest_upload": "Manifest JSON or export ZIP",
        "replay": "Replay manifest",
        "model_setup": "Local model setup",
        "model_confirm": "I reviewed the download size and license.",
        "model_prepare": "Prepare local model",
        "model_lite": "Lite · about 7.3 GB",
        "model_quality": "Quality · about 9.4 GB",
        "model_plan_lite": "Lite uses SDXL Base + the small SDXL Canny ControlNet. Files stay in the AIsketcher local cache. Review both model licenses before downloading.",
        "model_plan_quality": "Quality uses SDXL Base + the full SDXL Canny ControlNet. Files stay in the AIsketcher local cache. Review both model licenses before downloading.",
        "overrides": "Advanced overrides active",
        "clear": "Reset",
        "unavailable": "Guided Sample is unavailable because its bundled manifest or asset verification failed.",
        "ready": "Ready. Upload a sketch to begin.",
        "direction_label": "Direction {index}",
        "badge_most_distinct": "most distinct",
        "badge_closest_structure": "closest structure",
        "badge_cleanest_edges": "cleanest edges",
        "recommendation_heading": "Why this direction",
        "recommendation_detail": "Select a result to inspect its recorded structure and seed.",
        "seed_label": "seed",
        "status_generated": "Generated {count} recorded directions.",
        "status_guided": "Guided Sample loaded · provenance: {provenance}. Model controls are locked.",
        "status_selected": "Direction {index} selected.",
        "status_variation": "Created a new variation branch.",
        "status_replay": "Strict replay completed.",
        "status_export": "Export is ready.",
        "guide_body": """## From sketch to a reproducible study\n\n1. **Prepare** — upload a sketch; orientation and metadata are normalized locally.\n2. **Explore** — generate distinct seeded directions from one creative brief.\n3. **Pick & vary** — select a direction and keep its structure locked.\n4. **Export & replay** — preserve the actual manifest and lineage instead of relying on screenshots.\n\n### Models and provenance\n\nGuided Sample appears only when its source, candidate images, and real manifest are present. External reference images are never presented as AIsketcher output. Local model downloads require an explicit size-and-license confirmation; this app never asks for tokens or arbitrary model URLs.\n\nThe app binds to `127.0.0.1`, disables public sharing, validates uploads, and processes one generation at a time.""",
    },
    "ko": {
        "simple": "심플",
        "advanced": "고급 설정",
        "guide": "가이드",
        "headline": "한 장의 스케치를\n디자인 계보로.",
        "subhead": "네 방향을 탐색하고 하나를 선택한 뒤, 맥락을 잃지 않고 발전시키세요.",
        "sketch": "1  스케치",
        "upload": "스케치 업로드",
        "brief": "2  크리에이티브 브리프",
        "brief_placeholder": "분위기, 재질, 색상과 사용 목적을 적어주세요.",
        "profile": "작업 유형",
        "structure": "3  구조 유지",
        "loose": "자유롭게",
        "balanced": "균형 있게",
        "faithful": "충실하게",
        "explore": "방향 탐색하기",
        "guided": "가이드 샘플 체험",
        "source": "입력 (스케치)",
        "selected": "결과 (선택한 방향)",
        "directions": "디자인 방향",
        "directions_help": "스케치와 브리프에서 생성한 서로 다른 방향입니다.",
        "empty": "스케치를 추가하거나, 모델 다운로드 없이 번들 가이드 샘플을 열어보세요.",
        "refine": "이 방향 발전시키기",
        "again": "다른 방향 탐색",
        "export": "완성 및 내보내기",
        "tip": "팁: 구조를 잠근 채 선택한 방향을 발전시킬 수 있습니다.",
        "preset": "프리셋",
        "seed_plan": "시드 계획",
        "auto_seeds": "자동 (서로 다른 시드)",
        "locked_seed": "고정 시드 (같은 시작점)",
        "custom_seeds": "직접 입력",
        "locked_seed_placeholder": "예: 6764547109648557242",
        "custom_seed_placeholder": "예: 12345, 67890, 13579, 24680",
        "outputs": "출력 개수",
        "canny": "Canny 컨트롤",
        "canny_info": "현재 SDXL 프리셋에 필요한 설정입니다.",
        "steps": "스텝",
        "guidance": "가이던스",
        "variation": "변형 강도",
        "locks": "변형 잠금",
        "lock_structure": "구조 잠금",
        "manifest": "매니페스트와 재현",
        "manifest_upload": "매니페스트 JSON 또는 내보내기 ZIP",
        "replay": "매니페스트 재현",
        "model_setup": "로컬 모델 설정",
        "model_confirm": "다운로드 용량과 라이선스를 확인했습니다.",
        "model_prepare": "로컬 모델 준비",
        "model_lite": "Lite · 약 7.3 GB",
        "model_quality": "Quality · 약 9.4 GB",
        "model_plan_lite": "Lite는 SDXL Base와 소형 SDXL Canny ControlNet을 사용합니다. 파일은 AIsketcher 로컬 캐시에 저장됩니다. 다운로드 전에 두 모델의 라이선스를 확인하세요.",
        "model_plan_quality": "Quality는 SDXL Base와 전체 SDXL Canny ControlNet을 사용합니다. 파일은 AIsketcher 로컬 캐시에 저장됩니다. 다운로드 전에 두 모델의 라이선스를 확인하세요.",
        "overrides": "고급 설정 적용 중",
        "clear": "초기화",
        "unavailable": "가이드 샘플을 열 수 없습니다. 번들 매니페스트 또는 에셋 검증에 실패했습니다.",
        "ready": "준비되었습니다. 스케치를 업로드하세요.",
        "direction_label": "방향 {index}",
        "badge_most_distinct": "가장 차별화됨",
        "badge_closest_structure": "구조 유사도 최고",
        "badge_cleanest_edges": "가장 깔끔한 윤곽",
        "recommendation_heading": "이 방향의 기술적 특징",
        "recommendation_detail": "결과를 선택하면 기록된 구조와 시드를 확인할 수 있습니다.",
        "seed_label": "시드",
        "status_generated": "기록 가능한 방향 {count}개를 생성했습니다.",
        "status_guided": "가이드 샘플을 열었습니다 · 출처: {provenance}. 모델 설정은 잠겼습니다.",
        "status_selected": "{index}번 방향을 선택했습니다.",
        "status_variation": "새 변형 브랜치를 만들었습니다.",
        "status_replay": "엄격 재현을 완료했습니다.",
        "status_export": "내보내기 파일이 준비되었습니다.",
        "guide_body": """## 스케치에서 재현 가능한 스터디까지\n\n1. **준비** — 스케치를 올리면 방향과 메타데이터를 로컬에서 정규화합니다.\n2. **탐색** — 하나의 브리프로 서로 다른 시드의 방향을 만듭니다.\n3. **선택과 변형** — 마음에 드는 방향을 고르고 구조를 잠가 발전시킵니다.\n4. **내보내기와 재현** — 스크린샷이 아니라 실제 매니페스트와 계보를 보존합니다.\n\n### 모델과 출처\n\n가이드 샘플은 원본, 후보 이미지, 실제 매니페스트가 모두 있을 때만 표시됩니다. 외부 참고 이미지는 AIsketcher 출력으로 표시하지 않습니다. 로컬 모델은 용량과 라이선스를 명시적으로 확인한 뒤 다운로드하며, 이 앱은 토큰이나 임의 모델 URL을 요구하지 않습니다.\n\n앱은 `127.0.0.1`에만 바인딩하고 공개 공유를 끄며 업로드를 검증하고 한 번에 하나의 생성 작업만 처리합니다.""",
    },
}


def normalize_language(language: str | None) -> str:
    """Return a supported language code."""

    return language if language in SUPPORTED_LANGUAGES else "en"


def text(language: str | None, key: str) -> str:
    """Look up a translated string, falling back to English."""

    language = normalize_language(language)
    return _TEXT[language].get(key, _TEXT["en"].get(key, key))


def structure_choices(language: str | None) -> list[tuple[str, str]]:
    """Localized labels with stable values for the core API."""

    return [
        (text(language, "loose"), "loose"),
        (text(language, "balanced"), "balanced"),
        (text(language, "faithful"), "faithful"),
    ]


def navigation_choices(language: str | None) -> list[tuple[str, str]]:
    """Localized navigation labels with stable view identifiers."""

    return [
        (text(language, "simple"), "simple"),
        (text(language, "advanced"), "advanced"),
        (text(language, "guide"), "guide"),
    ]
