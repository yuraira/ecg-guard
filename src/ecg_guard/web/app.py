"""ECG Guard research demonstration web application."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from ecg_guard.data.prepare_ptbxl import LEAD_NAMES, load_waveform
from ecg_guard.inference.predict_record import (
    DEFAULT_PROTOCOL_PATH,
    RESEARCH_WARNING,
    load_inference_protocol,
    load_locked_model,
    predict_waveform,
    resolve_device,
)
from ecg_guard.web.presentation import (
    ACTION_LABELS_KO,
    create_ecg_figure,
    create_probability_figure,
    create_synthetic_demo_waveform,
    prediction_table,
    quality_flag_text,
    save_uploaded_record,
)


DEFAULT_CHECKPOINT_PATH = Path(
    os.environ.get(
        "ECG_GUARD_CHECKPOINT",
        "outputs/baseline/best_model.pt",
    )
)
DEMO_RECORD_PATH = os.environ.get("ECG_GUARD_DEMO_RECORD")
SYNTHETIC_DEMO_ENABLED = os.environ.get(
    "ECG_GUARD_ENABLE_SYNTHETIC_DEMO",
    "",
).strip().lower() in {"1", "true", "yes"}
DATA_POLICY_URL = (
    "https://github.com/yuraira/ecg-guard/blob/main/"
    "DATA_HANDLING_POLICY.md"
)
UPLOAD_CONFIRMATION = (
    "업로드 파일이 공개되었거나 적절하게 비식별 처리된 연구 데이터이며, "
    "실제 환자의 진료·진단·치료 목적으로 사용하지 않음을 확인합니다."
)


@st.cache_resource(show_spinner=False)
def load_bundle(
    checkpoint_path: str,
    checkpoint_modified_ns: int,
    requested_device: str,
) -> tuple[Any, Any, dict[str, Any], Any, str]:
    """Cache the locked model while invalidating when its file changes."""
    del checkpoint_modified_ns
    protocol = load_inference_protocol(DEFAULT_PROTOCOL_PATH)
    device = resolve_device(requested_device)
    model, normalization, digest = load_locked_model(
        Path(checkpoint_path),
        protocol,
        device,
    )
    return model, normalization, protocol, device, digest


def analyze_waveform(waveform: Any, record_id: str) -> dict[str, Any]:
    """Run the shared inference path and add non-identifying provenance."""
    checkpoint = DEFAULT_CHECKPOINT_PATH
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "동결된 체크포인트를 찾을 수 없습니다. "
            "ECG_GUARD_CHECKPOINT 환경 변수를 설정해 주세요."
        )
    model, normalization, protocol, device, digest = load_bundle(
        str(checkpoint.resolve()),
        checkpoint.stat().st_mtime_ns,
        os.environ.get("ECG_GUARD_DEVICE", "auto"),
    )
    report = predict_waveform(
        waveform,
        model,
        normalization,
        protocol,
        device,
    )
    report["input"] = {
        "record_id": record_id,
        "format": "WFDB",
        "sampling_frequency_hz": 100,
        "shape": list(waveform.shape),
        "lead_order": list(LEAD_NAMES),
        "amplitude_unit": "mV",
    }
    report["provenance"] = {
        "checkpoint_sha256": digest,
        "protocol_file": DEFAULT_PROTOCOL_PATH.name,
    }
    return report


def clear_session_analysis() -> None:
    """Remove analyzed data and reset the uploader widget for this session."""
    st.session_state.pop("analysis", None)
    generation = int(st.session_state.get("upload_generation", 0))
    st.session_state["upload_generation"] = generation + 1


def render_result(report: dict[str, Any], waveform: Any) -> None:
    """Render one result with review reasons and limitations in view."""
    if report.get("input", {}).get("synthetic"):
        st.warning(
            "합성 UI 샘플의 출력입니다. 모델 성능 사례나 임상적으로 유효한 "
            "ECG 결과로 해석할 수 없습니다.",
            icon="🧪",
        )
    action = report["routing"]["action"]
    action_label = ACTION_LABELS_KO[action]
    if action == "auto_result":
        st.success(f"라우팅 결과 · {action_label}", icon="✅")
    else:
        st.warning(f"라우팅 결과 · {action_label}", icon="⚠️")

    first, second, third, fourth = st.columns(4)
    first.metric("모델 버전", report["model_version"])
    second.metric(
        "결정 경계 불확실성",
        f"{report['uncertainty']['decision_uncertainty']:.3f}",
        help="높을수록 잠긴 클래스 임계값에 가깝습니다.",
    )
    third.metric(
        "기술 품질 점수",
        f"{report['technical_quality']['technical_quality_score']:.1f}/100",
        help="임상적 신호 품질 정답이나 오류 확률이 아닙니다.",
    )
    fourth.metric(
        "분류 표시",
        (
            "판단 보류"
            if report["routing"]["classification_withheld"]
            else "연구용 표시"
        ),
    )

    if report["routing"]["classification_withheld"]:
        st.info(
            "검증 데이터에서 고정한 불확실성 기준을 넘었습니다. "
            "클래스 결과를 확정 판정으로 사용하지 말고 재검토하세요."
        )
    if report["routing"]["technical_review_recommended"]:
        flags = report["technical_quality"]["review_flags"]
        st.info(f"기술 검토 사유: {quality_flag_text(flags)}")

    probability_tab, waveform_tab, details_tab = st.tabs(
        ["예측 확률", "12유도 파형", "품질·출처"]
    )
    with probability_tab:
        probability_figure = create_probability_figure(report)
        st.pyplot(probability_figure, width="stretch")
        plt.close(probability_figure)
        table = prediction_table(report)
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "보정 확률": st.column_config.NumberColumn(format="%.4f"),
                "임계값": st.column_config.NumberColumn(format="%.4f"),
                "경계 불확실성": st.column_config.NumberColumn(
                    format="%.4f"
                ),
            },
        )
    with waveform_tab:
        waveform_figure = create_ecg_figure(waveform)
        st.pyplot(waveform_figure, width="stretch")
        plt.close(waveform_figure)
    with details_tab:
        st.markdown("#### 기술 품질")
        st.write(
            f"상태: `{report['technical_quality']['technical_quality_status']}`"
        )
        st.write(
            "검토 항목: "
            + quality_flag_text(
                report["technical_quality"]["review_flags"]
            )
        )
        st.caption(report["technical_quality"]["interpretation"])
        st.markdown("#### 추론 출처")
        st.code(
            "\n".join(
                (
                    f"model: {report['model_version']}",
                    "checkpoint_sha256: "
                    f"{report['provenance']['checkpoint_sha256']}",
                    "protocol: "
                    f"{report['provenance']['protocol_file']}",
                )
            ),
            language="text",
        )

    export = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    st.download_button(
        "결과 JSON 내려받기",
        data=export,
        file_name=f"{report['input']['record_id']}_ecg_guard.json",
        mime="application/json",
        width="stretch",
    )


def main() -> None:
    st.set_page_config(
        page_title="ECG Guard",
        page_icon="♡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2.2rem;}
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dce6e8;
            border-radius: 14px;
            padding: 0.85rem 1rem;
        }
        [data-testid="stSidebar"] {border-right: 1px solid #dce6e8;}
        h1, h2, h3 {letter-spacing: -0.025em;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("ECG Guard")
        st.caption("baseline v1 · 신뢰도 중심 ECG AI 데모")
        checkpoint_ready = DEFAULT_CHECKPOINT_PATH.is_file()
        if checkpoint_ready:
            st.success("동결 체크포인트 준비됨")
        else:
            st.error("동결 체크포인트 필요")
        st.markdown("#### 입력 요구사항")
        st.write("100Hz · 10초 · 12유도 · mV")
        st.caption(
            "업로드 파일은 분석 중 임시 폴더에만 저장되며 앱이 별도로 "
            "보관하지 않습니다."
        )
        with st.expander("모델의 주요 한계"):
            st.write(
                "단일 공개 데이터셋과 단일 학습 seed 결과입니다. "
                "외부 병원·장비·전향적 임상 데이터에서 검증하지 않았습니다."
            )

    st.title("ECG 결과보다 먼저, 신뢰할 조건을 확인합니다")
    st.write(
        "12유도 파형과 다섯 진단 상위군의 연구용 분류 결과를 "
        "신호 품질·결정 경계 불확실성과 함께 살펴보세요."
    )
    st.warning(RESEARCH_WARNING, icon="⚠️")

    upload_tab, sample_tab = st.tabs(["WFDB 파일 업로드", "로컬 데모 샘플"])
    with upload_tab:
        st.subheader("한 건의 WFDB 레코드 분석")
        st.caption(
            "동일 레코드의 .hea 파일 1개와 헤더가 참조하는 .dat 파일을 "
            "함께 선택하세요. 최대 전체 50MB입니다."
        )
        st.info(
            "실제 환자 파일은 업로드할 수 없습니다. 공개되었거나 적절하게 "
            "비식별 처리된 연구 데이터만 사용하세요."
        )
        st.markdown(f"[데모 데이터 처리 정책 전문]({DATA_POLICY_URL})")
        upload_generation = int(
            st.session_state.get("upload_generation", 0)
        )
        policy_confirmed = st.checkbox(
            UPLOAD_CONFIRMATION,
            key=f"upload-policy-confirmed-{upload_generation}",
        )
        uploaded_files = st.file_uploader(
            "WFDB 파일 선택",
            type=["hea", "dat"],
            accept_multiple_files=True,
            help="표준 12유도, 100Hz, 유도당 1,000표본만 지원합니다.",
            key=f"wfdb-upload-{upload_generation}",
        )
        if st.button(
            "업로드 ECG 분석",
            type="primary",
            disabled=(
                not uploaded_files
                or not checkpoint_ready
                or not policy_confirmed
            ),
            width="stretch",
        ):
            try:
                with st.spinner("입력과 체크포인트를 검증하고 있습니다..."):
                    with tempfile.TemporaryDirectory(
                        prefix="ecg_guard_"
                    ) as temporary:
                        header = save_uploaded_record(
                            uploaded_files,
                            Path(temporary),
                        )
                        waveform = load_waveform(
                            header.stem,
                            header.parent,
                        )
                        report = analyze_waveform(
                            waveform,
                            "uploaded-record",
                        )
                st.session_state["analysis"] = {
                    "report": report,
                    "waveform": waveform,
                }
                st.session_state["upload_generation"] = (
                    upload_generation + 1
                )
                st.rerun()
            except (ValueError, FileNotFoundError, RuntimeError) as error:
                st.error(str(error))

    with sample_tab:
        st.subheader("사전에 지정한 비식별 공개 샘플")
        if not DEMO_RECORD_PATH and not SYNTHETIC_DEMO_ENABLED:
            st.info(
                "샘플을 사용하려면 ECG_GUARD_DEMO_RECORD 환경 변수에 "
                "WFDB 레코드 기본 경로를 설정하거나 합성 데모를 활성화하세요."
            )
        elif DEMO_RECORD_PATH:
            sample_path = Path(DEMO_RECORD_PATH)
            st.code(sample_path.name, language="text")
            if st.button(
                "데모 샘플 분석",
                type="primary",
                disabled=not checkpoint_ready,
                width="stretch",
            ):
                try:
                    with st.spinner("샘플을 분석하고 있습니다..."):
                        if sample_path.suffix.lower() in {".hea", ".dat"}:
                            sample_path = sample_path.with_suffix("")
                        waveform = load_waveform(
                            sample_path.name,
                            sample_path.parent,
                        )
                        report = analyze_waveform(
                            waveform,
                            sample_path.name,
                        )
                    st.session_state["analysis"] = {
                        "report": report,
                        "waveform": waveform,
                    }
                except (ValueError, FileNotFoundError, RuntimeError) as error:
                    st.error(str(error))
        else:
            st.info(
                "이 파형은 화면 동작 확인을 위해 코드로 생성한 합성 신호입니다. "
                "실제 환자 기록이나 임상적으로 검증된 ECG 예시가 아닙니다."
            )
            if st.button(
                "합성 샘플로 전체 흐름 보기",
                type="primary",
                disabled=not checkpoint_ready,
                width="stretch",
            ):
                try:
                    with st.spinner("합성 샘플을 분석하고 있습니다..."):
                        waveform = create_synthetic_demo_waveform()
                        report = analyze_waveform(
                            waveform,
                            "synthetic-ui-demo",
                        )
                        report["input"]["synthetic"] = True
                    st.session_state["analysis"] = {
                        "report": report,
                        "waveform": waveform,
                    }
                except (ValueError, FileNotFoundError, RuntimeError) as error:
                    st.error(str(error))

    analysis = st.session_state.get("analysis")
    if analysis:
        st.divider()
        st.header("분석 결과")
        render_result(analysis["report"], analysis["waveform"])
        st.button(
            "세션의 분석 결과 삭제",
            on_click=clear_session_analysis,
            width="stretch",
            help="메모리에 유지 중인 파형과 분석 결과를 현재 세션에서 제거합니다.",
        )
    elif not checkpoint_ready:
        st.info(
            "먼저 README의 학습 절차로 baseline v1 체크포인트를 준비하거나 "
            "ECG_GUARD_CHECKPOINT에 해당 파일 경로를 지정하세요."
        )

    st.divider()
    st.caption(
        "ECG Guard는 연구·교육용 오픈소스 프로젝트입니다. 출력 확률은 "
        "환자별 질병 위험도나 의료적 진단을 의미하지 않습니다."
    )


if __name__ == "__main__":
    main()
