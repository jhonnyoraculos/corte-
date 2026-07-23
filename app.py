from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging

import streamlit as st

from src.contour_processing import contour_summary, extract_contours
from src.exporters.dxf_exporter import DXFExportOptions, generate_dxf_bytes
from src.exporters.mpr_exporter import TemplateMPRPostProcessor
from src.exporters.mpr_template import analyze_mpr_reference
from src.geometry import (
    align_lower_left,
    bounding_box,
    center_at_origin,
    dimensions,
    offset_contour,
    pixels_per_mm_from_points,
    pixels_per_mm_from_width,
    reverse_contour,
    transform_contours,
)
from src.image_processing import (
    decode_image,
    image_quality_warnings,
    process_image,
)
from src.models import (
    CNCProject,
    Calibration,
    MachineProfile,
    ProcessingParameters,
    ValidationSeverity,
)
from src.preview import build_vector_preview
from src.utils.files import (
    UploadValidationError,
    load_json_upload,
    safe_filename,
    validate_image_upload,
)
from src.validators import has_critical_errors, validate_project


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("cnc_vectorizer.app")

st.set_page_config(
    page_title="Vetor CNC a partir de imagem",
    page_icon="📐",
    layout="wide",
)

SAFETY_NOTICE = (
    "Confira o arquivo no software oficial da máquina e faça uma simulação antes "
    "de qualquer operação real. O aplicativo não substitui a validação técnica "
    "do operador."
)

STAGES = [
    "1 — Enviar imagem",
    "2 — Calibração e dimensões",
    "3 — Tratamento da imagem",
    "4 — Extração dos contornos",
    "5 — Edição geométrica",
    "6 — Prévia vetorial",
    "7 — Configurações da ferramenta",
    "8 — Validar e exportar",
]


@st.cache_data(show_spinner=False)
def cached_processing(image_content: bytes, params_json: str):
    image_rgb = decode_image(image_content)
    params = ProcessingParameters(**json.loads(params_json))
    return process_image(image_rgb, params)


def initialize_state() -> None:
    if "project" not in st.session_state:
        st.session_state.project = CNCProject()
    st.session_state.setdefault("image_content", None)
    st.session_state.setdefault("image_digest", None)
    st.session_state.setdefault("image_rgb", None)
    st.session_state.setdefault("extraction_signature", None)
    st.session_state.setdefault("mpr_reference", None)
    st.session_state.setdefault("mpr_program_template", "")


def clear_project() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def processing_signature(project: CNCProject) -> str:
    payload = {
        "processing": asdict(project.processing),
        "scale": project.calibration.pixels_per_mm,
        "image": st.session_state.image_digest,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def get_processing_result():
    content = st.session_state.image_content
    if content is None:
        return None
    params_json = json.dumps(
        asdict(st.session_state.project.processing), sort_keys=True
    )
    return cached_processing(content, params_json)


def commit_geometry(
    contours,
    operation: str,
    details: dict[str, object] | None = None,
) -> None:
    project: CNCProject = st.session_state.project
    project.contours = contours
    project.transformations.append(
        {"operation": operation, "details": details or {}}
    )
    project.validation_issues = validate_project(project)


def render_issue_list(issues) -> None:
    icons = {
        ValidationSeverity.ERROR: "🔴",
        ValidationSeverity.WARNING: "🟠",
        ValidationSeverity.INFO: "🔵",
    }
    if not issues:
        st.success("Nenhum problema encontrado.")
        return
    for issue in issues:
        suffix = f" — {issue.contour_id}" if issue.contour_id else ""
        st.write(f"{icons[issue.severity]} **{issue.severity.value.title()}**{suffix}: {issue.message}")


def render_project_sidebar(project: CNCProject) -> str:
    st.sidebar.title("Vetor CNC")
    stage = st.sidebar.radio("Etapas do projeto", STAGES)
    st.sidebar.divider()
    project.name = st.sidebar.text_input(
        "Nome do projeto", value=project.name, help="Usado nos arquivos exportados."
    )
    st.sidebar.download_button(
        "Baixar projeto JSON",
        data=project.to_json().encode("utf-8"),
        file_name=f"{safe_filename(project.name, 'projeto')}.cnc.json",
        mime="application/json",
        use_container_width=True,
    )
    imported_project = st.sidebar.file_uploader(
        "Carregar projeto JSON",
        type=["json"],
        key="project_import",
        help="A imagem original não é armazenada no projeto.",
    )
    if imported_project is not None and st.sidebar.button(
        "Importar projeto", use_container_width=True
    ):
        try:
            data = load_json_upload(imported_project.getvalue())
            imported = CNCProject.from_dict(data)
            imported_digest = imported.image_metadata.get("sha256")
            if imported_digest != st.session_state.image_digest:
                st.session_state.image_content = None
                st.session_state.image_digest = None
                st.session_state.image_rgb = None
            st.session_state.project = imported
            st.session_state.extraction_signature = None
            st.sidebar.success("Projeto importado.")
            st.rerun()
        except (UploadValidationError, ValueError, TypeError) as exc:
            st.sidebar.error(str(exc))
    col_reset, col_clear = st.sidebar.columns(2)
    if col_reset.button(
        "Restaurar filtros",
        help="Restaura somente os parâmetros de processamento.",
        use_container_width=True,
    ):
        project.processing = ProcessingParameters()
        st.session_state.extraction_signature = None
        st.rerun()
    if col_clear.button(
        "Limpar tudo",
        help="Remove imagem, contornos e configurações da sessão.",
        use_container_width=True,
    ):
        clear_project()
    return stage


def stage_upload(project: CNCProject) -> None:
    st.header("1 — Enviar imagem")
    st.write(
        "Envie uma vista ortogonal, com bom contraste, pouca perspectiva e a peça "
        "inteira dentro do quadro."
    )
    uploaded = st.file_uploader(
        "Imagem da peça",
        type=["png", "jpg", "jpeg", "bmp"],
        help="PNG, JPG, JPEG ou BMP; limite de 20 MB.",
        key="image_upload",
    )
    if uploaded is not None:
        content = uploaded.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        if digest != st.session_state.image_digest:
            try:
                metadata = validate_image_upload(uploaded.name, content)
                image_rgb = decode_image(content)
                st.session_state.image_content = content
                st.session_state.image_rgb = image_rgb
                st.session_state.image_digest = digest
                st.session_state.extraction_signature = None
                project.image_metadata = {
                    **metadata,
                    "sha256": digest,
                }
                project.contours = []
                logger.info("Imagem validada: %s", metadata["filename"])
            except UploadValidationError as exc:
                st.error(str(exc))
                return
    if st.session_state.image_rgb is None:
        st.info("Nenhuma imagem válida foi carregada.")
        return
    metadata = project.image_metadata
    left, right = st.columns([3, 2])
    left.image(
        st.session_state.image_rgb,
        caption=metadata.get("filename", "Imagem original"),
        use_container_width=True,
    )
    right.metric(
        "Resolução",
        f"{metadata.get('width_px', 0)} × {metadata.get('height_px', 0)} px",
    )
    right.metric("Tamanho", f"{metadata.get('size_bytes', 0) / 1024:.1f} KB")
    for warning in image_quality_warnings(st.session_state.image_rgb):
        right.warning(warning)
    right.info(
        "A imagem só permanece na sessão atual. O JSON do projeto salva metadados, "
        "escala, filtros e vetores, não a imagem completa."
    )


def stage_calibration(project: CNCProject) -> None:
    st.header("2 — Calibração e dimensões")
    if st.session_state.image_rgb is None:
        st.info("Envie uma imagem na etapa 1.")
        return
    image_height, image_width = st.session_state.image_rgb.shape[:2]
    method = st.radio(
        "Método de calibração",
        ["Largura conhecida", "Dois pontos de referência"],
        horizontal=True,
    )
    try:
        if method == "Largura conhecida":
            st.caption(
                "No MVP, a largura em pixels corresponde à largura completa da imagem. "
                "Recorte a imagem exatamente nos limites da referência ou da peça."
            )
            width_mm = st.number_input(
                "Largura real correspondente à imagem (mm)",
                min_value=0.001,
                value=float(project.calibration.reference_width_mm or 500.0),
                step=1.0,
            )
            height_mm = st.number_input(
                "Altura real opcional (mm; 0 para ignorar)",
                min_value=0.0,
                value=float(project.calibration.reference_height_mm or 0.0),
                step=1.0,
            )
            if st.button("Aplicar calibração pela largura", type="primary"):
                scale = pixels_per_mm_from_width(image_width, width_mm)
                project.calibration = Calibration(
                    method="known_width",
                    pixels_per_mm=scale,
                    reference_width_mm=width_mm,
                    reference_height_mm=height_mm or None,
                )
                project.contours = []
                st.session_state.extraction_signature = None
                if height_mm:
                    scale_height = image_height / height_mm
                    difference = abs(scale - scale_height) / scale * 100
                    if difference > 2:
                        st.warning(
                            f"A altura informada resulta em escala {difference:.1f}% "
                            "diferente; a largura foi usada como referência principal."
                        )
                st.success("Escala aplicada. Refaça a extração dos contornos.")
        else:
            st.caption("Informe X e Y em pixels a partir do canto superior esquerdo.")
            col1, col2 = st.columns(2)
            default_p1 = project.calibration.point_1_px or (0.0, 0.0)
            default_p2 = project.calibration.point_2_px or (
                float(image_width),
                0.0,
            )
            with col1:
                st.subheader("Ponto 1")
                x1 = st.number_input("X1 (px)", 0.0, float(image_width), float(default_p1[0]))
                y1 = st.number_input("Y1 (px)", 0.0, float(image_height), float(default_p1[1]))
            with col2:
                st.subheader("Ponto 2")
                x2 = st.number_input("X2 (px)", 0.0, float(image_width), float(default_p2[0]))
                y2 = st.number_input("Y2 (px)", 0.0, float(image_height), float(default_p2[1]))
            distance_mm = st.number_input(
                "Distância real entre os pontos (mm)",
                min_value=0.001,
                value=float(project.calibration.reference_distance_mm or 500.0),
            )
            if st.button("Aplicar calibração pelos pontos", type="primary"):
                scale = pixels_per_mm_from_points(
                    (x1, y1), (x2, y2), distance_mm
                )
                project.calibration = Calibration(
                    method="two_points",
                    pixels_per_mm=scale,
                    point_1_px=(x1, y1),
                    point_2_px=(x2, y2),
                    reference_distance_mm=distance_mm,
                )
                project.contours = []
                st.session_state.extraction_signature = None
                st.success("Escala aplicada. Refaça a extração dos contornos.")
    except ValueError as exc:
        st.error(str(exc))

    if project.calibration.is_defined:
        scale = project.calibration.pixels_per_mm or 1.0
        col1, col2, col3 = st.columns(3)
        col1.metric("Escala", f"{scale:.6f} px/mm")
        col2.metric("Largura da imagem", f"{image_width / scale:.3f} mm")
        col3.metric("Altura da imagem", f"{image_height / scale:.3f} mm")
        if project.selected_contours:
            width, height = dimensions(project.selected_contours)
            st.info(f"Desenho atual: {width:.3f} × {height:.3f} mm.")
    else:
        st.warning("Nenhuma escala foi assumida. Informe uma medida real para continuar.")


def render_processing_controls(params: ProcessingParameters) -> None:
    with st.sidebar.expander("Controles de imagem", expanded=True):
        params.convert_grayscale = st.checkbox(
            "Converter para tons de cinza", value=params.convert_grayscale
        )
        params.invert = st.checkbox("Inverter imagem", value=params.invert)
        params.gaussian_blur = st.checkbox(
            "Desfoque gaussiano", value=params.gaussian_blur
        )
        params.gaussian_kernel = st.slider(
            "Kernel gaussiano", 1, 15, params.gaussian_kernel, step=2
        )
        params.denoise = st.checkbox("Remover ruído", value=params.denoise)
        modes = ["manual", "otsu", "adaptativo"]
        params.threshold_mode = st.selectbox(
            "Tipo de limiar",
            modes,
            index=modes.index(params.threshold_mode),
        )
        if params.threshold_mode == "manual":
            params.manual_threshold = st.slider(
                "Limiar manual", 0, 255, params.manual_threshold
            )
        elif params.threshold_mode == "adaptativo":
            params.adaptive_block_size = st.slider(
                "Bloco adaptativo", 3, 51, params.adaptive_block_size, step=2
            )
            params.adaptive_c = st.slider(
                "Constante adaptativa", -20, 20, params.adaptive_c
            )
        params.use_canny = st.checkbox(
            "Usar Canny para os contornos", value=params.use_canny
        )
        params.canny_min = st.slider("Canny mínimo", 0, 255, params.canny_min)
        params.canny_max = st.slider("Canny máximo", 0, 255, params.canny_max)
        params.morphology_kernel = st.slider(
            "Kernel morfológico", 1, 15, params.morphology_kernel, step=2
        )
        params.closing_iterations = st.slider(
            "Fechamento", 0, 5, params.closing_iterations
        )
        params.opening_iterations = st.slider(
            "Abertura", 0, 5, params.opening_iterations
        )
        params.dilation_iterations = st.slider(
            "Dilatação", 0, 5, params.dilation_iterations
        )
        params.erosion_iterations = st.slider(
            "Erosão", 0, 5, params.erosion_iterations
        )


def stage_processing(project: CNCProject) -> None:
    st.header("3 — Tratamento da imagem")
    if st.session_state.image_content is None:
        st.info("Envie uma imagem na etapa 1.")
        return
    render_processing_controls(project.processing)
    with st.spinner("Processando a imagem..."):
        result = get_processing_result()
    if result is None:
        return
    if result.threshold_value is not None:
        st.caption(f"Limiar calculado/aplicado: {result.threshold_value:.1f}")
    col1, col2 = st.columns(2)
    col1.image(result.original_rgb, caption="Original", use_container_width=True)
    col2.image(result.grayscale, caption="Tons de cinza", use_container_width=True)
    col3, col4 = st.columns(2)
    col3.image(result.binary, caption="Imagem binária", use_container_width=True)
    col4.image(result.edges, caption="Bordas Canny", use_container_width=True)
    st.info(
        "A fonte atual para extração é "
        + ("Canny." if project.processing.use_canny else "a imagem binária.")
    )


def render_contour_controls(params: ProcessingParameters) -> None:
    with st.sidebar.expander("Controles de contorno", expanded=True):
        params.ignore_small_noise = st.checkbox(
            "Ignorar pequenos ruídos", value=params.ignore_small_noise
        )
        params.min_area_mm2 = st.number_input(
            "Área mínima (mm²)", min_value=0.0, value=params.min_area_mm2
        )
        params.min_perimeter_mm = st.number_input(
            "Perímetro mínimo (mm)", min_value=0.0, value=params.min_perimeter_mm
        )
        params.keep_largest_only = st.checkbox(
            "Manter somente a maior peça", value=params.keep_largest_only
        )
        params.keep_holes = st.checkbox(
            "Manter furos internos", value=params.keep_holes
        )
        params.close_gaps = st.checkbox(
            "Fechar pequenos espaços", value=params.close_gaps
        )
        params.closing_tolerance_mm = st.number_input(
            "Tolerância de fechamento (mm)",
            min_value=0.0,
            value=params.closing_tolerance_mm,
        )
        params.simplify = st.checkbox(
            "Simplificar contornos", value=params.simplify
        )
        params.simplify_tolerance_mm = st.number_input(
            "Tolerância de simplificação (mm)",
            min_value=0.0,
            value=params.simplify_tolerance_mm,
            step=0.01,
        )
        params.smooth = st.checkbox("Suavizar contornos", value=params.smooth)
        params.smoothing_window = st.slider(
            "Janela de suavização", 2, 9, params.smoothing_window
        )


def stage_contours(project: CNCProject) -> None:
    st.header("4 — Extração dos contornos")
    if st.session_state.image_content is None:
        st.info("Envie uma imagem na etapa 1.")
        return
    if not project.calibration.is_defined:
        st.warning("Defina a escala na etapa 2 antes de extrair.")
        return
    render_contour_controls(project.processing)
    current_signature = processing_signature(project)
    if (
        project.contours
        and st.session_state.extraction_signature != current_signature
    ):
        st.warning("Filtros ou escala mudaram. Extraia novamente para atualizar os vetores.")
    if st.button("Detectar contornos", type="primary"):
        try:
            result = get_processing_result()
            with st.spinner("Localizando e convertendo contornos..."):
                project.contours = extract_contours(
                    result.contour_source,
                    project.calibration.pixels_per_mm or 0.0,
                    project.processing,
                )
            st.session_state.extraction_signature = current_signature
            project.validation_issues = validate_project(project)
            if project.contours:
                st.success(f"{len(project.contours)} contorno(s) encontrado(s).")
            else:
                st.error(
                    "Nenhum contorno passou pelos filtros. Ajuste o limiar, inverta "
                    "a imagem ou reduza os mínimos."
                )
        except (ValueError, RuntimeError) as exc:
            st.error(f"Não foi possível extrair os contornos: {exc}")
    if not project.contours:
        return

    summaries = [contour_summary(contour) for contour in project.contours]
    edited = st.data_editor(
        summaries,
        hide_index=True,
        use_container_width=True,
        disabled=[
            "ID",
            "Tipo",
            "Área (mm²)",
            "Perímetro (mm)",
            "Pontos",
            "Fechado",
            "Nível",
            "Sentido",
        ],
        column_config={
            "Selecionado": st.column_config.CheckboxColumn("Usar", required=True)
        },
        key="contour_editor",
    )
    rows = edited.to_dict("records") if hasattr(edited, "to_dict") else edited
    selected_by_id = {row["ID"]: bool(row["Selecionado"]) for row in rows}
    for contour in project.contours:
        contour.selected = selected_by_id.get(contour.contour_id, contour.selected)

    delete_ids = st.multiselect(
        "Remover contornos definitivamente",
        [contour.contour_id for contour in project.contours],
    )
    if st.button("Remover escolhidos", disabled=not delete_ids):
        commit_geometry(
            [
                contour
                for contour in project.contours
                if contour.contour_id not in delete_ids
            ],
            "delete_contours",
            {"ids": delete_ids},
        )
        st.rerun()


def stage_geometry(project: CNCProject) -> None:
    st.header("5 — Edição geométrica")
    if not project.contours:
        st.info("Extraia contornos na etapa 4.")
        return
    selected_ids = [
        contour.contour_id for contour in project.contours if contour.selected
    ]
    if not selected_ids:
        st.warning("Selecione pelo menos um contorno na etapa 4.")
        return

    st.subheader("Contorno individual")
    col1, col2, col3 = st.columns(3)
    contour_id = col1.selectbox("Contorno", selected_ids)
    if col2.button("Inverter direção", use_container_width=True):
        changed = [
            reverse_contour(contour) if contour.contour_id == contour_id else contour
            for contour in project.contours
        ]
        commit_geometry(changed, "reverse_contour", {"id": contour_id})
        st.rerun()
    target = next(
        contour for contour in project.contours if contour.contour_id == contour_id
    )
    if col3.button(
        "Fechar contorno",
        disabled=target.closed,
        use_container_width=True,
    ):
        target.closed = True
        commit_geometry(project.contours, "close_contour", {"id": contour_id})
        st.rerun()

    st.subheader("Mover, espelhar e girar")
    col1, col2, col3 = st.columns(3)
    move_x = col1.number_input("Deslocamento X (mm)", value=0.0)
    move_y = col2.number_input("Deslocamento Y (mm)", value=0.0)
    rotation = col3.number_input("Rotação (graus)", value=0.0)
    mirror_h = col1.checkbox("Espelhar horizontalmente")
    mirror_v = col2.checkbox("Espelhar verticalmente")
    if st.button("Aplicar transformação"):
        changed_selected = transform_contours(
            [contour for contour in project.contours if contour.selected],
            translate_x=move_x,
            translate_y=move_y,
            rotate_degrees=rotation,
            mirror_horizontal=mirror_h,
            mirror_vertical=mirror_v,
        )
        replacements = {
            contour.contour_id: contour for contour in changed_selected
        }
        changed = [
            replacements.get(contour.contour_id, contour)
            for contour in project.contours
        ]
        commit_geometry(
            changed,
            "transform",
            {
                "x": move_x,
                "y": move_y,
                "rotation": rotation,
                "mirror_horizontal": mirror_h,
                "mirror_vertical": mirror_v,
            },
        )
        st.rerun()

    st.subheader("Definir a origem")
    col1, col2, col3 = st.columns(3)
    if col1.button("Inferior esquerdo em 0,0", use_container_width=True):
        commit_geometry(align_lower_left(project.contours), "align_lower_left")
        st.rerun()
    if col2.button("Centralizar em 0,0", use_container_width=True):
        commit_geometry(center_at_origin(project.contours), "center_at_origin")
        st.rerun()
    custom_x = col3.number_input("Origem personalizada X", value=0.0)
    custom_y = col3.number_input("Origem personalizada Y", value=0.0)
    if col3.button("Mover canto para origem", use_container_width=True):
        commit_geometry(
            align_lower_left(project.contours, (custom_x, custom_y)),
            "custom_origin",
            {"x": custom_x, "y": custom_y},
        )
        st.rerun()

    st.subheader("Compensação")
    compensation_mode = st.radio(
        "Valor da compensação",
        ["Raio da ferramenta", "Kerf", "Personalizado"],
        horizontal=True,
    )
    if compensation_mode == "Raio da ferramenta":
        amount = project.tool.radius_mm
        st.caption(f"Raio atual: {amount:.3f} mm.")
    elif compensation_mode == "Kerf":
        kerf = st.number_input("Largura do kerf (mm)", min_value=0.0, value=0.2)
        amount = kerf / 2
    else:
        amount = st.number_input("Deslocamento (mm)", min_value=0.0, value=0.1)
    side = st.radio(
        "Lado", ["Externa (aumentar)", "Interna (reduzir)"], horizontal=True
    )
    compensate_ids = st.multiselect(
        "Contornos que receberão a compensação",
        selected_ids,
        default=selected_ids,
    )
    if st.button("Aplicar compensação", disabled=not compensate_ids):
        signed_distance = amount if side.startswith("Externa") else -amount
        try:
            changed = []
            for contour in project.contours:
                changed.append(
                    offset_contour(contour, signed_distance)
                    if contour.contour_id in compensate_ids
                    else contour
                )
            commit_geometry(
                changed,
                "offset",
                {"distance_mm": signed_distance, "ids": compensate_ids},
            )
            st.rerun()
        except ValueError as exc:
            st.error(f"Compensação cancelada sem alterar o projeto: {exc}")

    project.validation_issues = validate_project(project)
    errors = [
        issue
        for issue in project.validation_issues
        if issue.severity == ValidationSeverity.ERROR
    ]
    if errors:
        st.error(f"A geometria atual possui {len(errors)} erro(s) de validação.")
    width, height = dimensions(project.selected_contours)
    min_x, min_y, max_x, max_y = bounding_box(project.selected_contours)
    st.caption(
        f"Caixa: X {min_x:.3f}…{max_x:.3f} mm, "
        f"Y {min_y:.3f}…{max_y:.3f} mm — {width:.3f} × {height:.3f} mm."
    )


def stage_preview(project: CNCProject) -> None:
    st.header("6 — Prévia vetorial")
    if not project.selected_contours:
        st.info("Selecione contornos antes de abrir a prévia.")
        return
    col1, col2, col3 = st.columns(3)
    show_points = col1.checkbox("Exibir pontos")
    show_compensation = col2.checkbox("Destacar compensação", value=True)
    show_ids = col3.checkbox("Exibir IDs", value=True)
    figure = build_vector_preview(
        project.contours,
        show_points=show_points,
        show_compensation=show_compensation,
        show_ids=show_ids,
    )
    st.plotly_chart(figure, use_container_width=True)
    width, height = dimensions(project.selected_contours)
    col1, col2, col3 = st.columns(3)
    col1.metric("Largura total", f"{width:.3f} mm")
    col2.metric("Altura total", f"{height:.3f} mm")
    col3.metric("Contornos", len(project.selected_contours))


def stage_tool(project: CNCProject) -> None:
    st.header("7 — Configurações da ferramenta")
    st.write(
        "Estes valores ficam no projeto. O DXF guarda a geometria; o MPR só recebe "
        "um valor quando o template fornecido comprova e solicita esse campo."
    )
    tool = project.tool
    col1, col2 = st.columns(2)
    tool.diameter_mm = col1.number_input(
        "Diâmetro da ferramenta (mm)", min_value=0.001, value=tool.diameter_mm
    )
    col2.metric("Raio calculado", f"{tool.radius_mm:.3f} mm")
    tool.cutting_depth_mm = col1.number_input(
        "Profundidade de corte (mm)", min_value=0.0, value=tool.cutting_depth_mm
    )
    tool.depth_per_pass_mm = col2.number_input(
        "Profundidade por passe (mm)", min_value=0.001, value=tool.depth_per_pass_mm
    )
    tool.feed_rate_mm_min = col1.number_input(
        "Velocidade de avanço (mm/min)", min_value=0.0, value=tool.feed_rate_mm_min
    )
    tool.plunge_rate_mm_min = col2.number_input(
        "Velocidade de mergulho (mm/min)", min_value=0.0, value=tool.plunge_rate_mm_min
    )
    tool.spindle_rpm = int(
        col1.number_input("Rotação (RPM)", min_value=0, value=tool.spindle_rpm)
    )
    tool.safety_margin_mm = col2.number_input(
        "Margem de segurança (mm)", min_value=0.0, value=tool.safety_margin_mm
    )
    directions = ["convencional", "concordante"]
    tool.cutting_direction = col1.selectbox(
        "Sentido de corte",
        directions,
        index=directions.index(tool.cutting_direction),
    )
    col2.text_input("Unidade", value="mm", disabled=True)
    project.material_thickness_mm = st.number_input(
        "Espessura do material/peça (mm)",
        min_value=0.001,
        value=project.material_thickness_mm,
    )


def render_dxf_export(project: CNCProject, issues) -> None:
    st.subheader("DXF")
    version = st.radio("Versão DXF", ["R2000", "R12"], horizontal=True)
    include_reference = st.checkbox("Incluir caixa e origem na camada REFERENCE", value=True)
    include_start = st.checkbox("Incluir pontos iniciais", value=True)
    warnings = [
        issue for issue in issues if issue.severity == ValidationSeverity.WARNING
    ]
    confirm = True
    if warnings:
        confirm = st.checkbox(
            f"Revisei e aceito exportar com {len(warnings)} aviso(s)."
        )
    blocked = has_critical_errors(issues) or not confirm
    if blocked:
        st.info("Resolva os erros e confirme os avisos para liberar o DXF.")
        return
    try:
        dxf_data = generate_dxf_bytes(
            project,
            DXFExportOptions(
                version=version,
                include_reference=include_reference,
                include_start_points=include_start,
            ),
        )
        st.download_button(
            f"Baixar DXF {version}",
            data=dxf_data,
            file_name=f"{safe_filename(project.name, 'projeto')}_{version.lower()}.dxf",
            mime="application/dxf",
            use_container_width=True,
        )
    except (ValueError, TypeError) as exc:
        st.error(f"Falha ao preparar o DXF: {exc}")


def render_mpr_export(project: CNCProject, geometry_issues) -> None:
    st.subheader("MPR — pós-processador experimental")
    st.warning(
        "MPR não é um padrão universal. Este aplicativo não inventa comandos: "
        "ele apenas substitui placeholders em uma cópia editável baseada em um "
        "arquivo real e em um perfil fornecido pelo operador."
    )
    profile_upload = st.file_uploader(
        "Perfil JSON da máquina",
        type=["json"],
        key="machine_profile_upload",
    )
    if profile_upload is not None and st.button("Carregar perfil MPR"):
        try:
            data = load_json_upload(profile_upload.getvalue(), max_bytes=1_000_000)
            project.machine_profile = MachineProfile.from_dict(data)
            project.validation_issues = validate_project(project)
            st.success(f"Perfil carregado: {project.machine_profile.profile_name}")
            st.rerun()
        except (UploadValidationError, TypeError, ValueError) as exc:
            st.error(f"Perfil rejeitado: {exc}")
    if project.machine_profile:
        st.info(f"Perfil ativo: {project.machine_profile.profile_name}")

    reference_upload = st.file_uploader(
        "Arquivo MPR real de referência",
        type=["mpr"],
        key="mpr_reference_upload",
        help="Somente texto, até 2 MB. O arquivo original não é modificado nem executado.",
    )
    if reference_upload is not None and st.button("Analisar referência MPR"):
        try:
            analysis = analyze_mpr_reference(reference_upload.getvalue())
            st.session_state.mpr_reference = analysis
            st.session_state.mpr_program_template = analysis.text
            st.success("Referência carregada em modo somente leitura.")
        except UploadValidationError as exc:
            st.error(f"Referência rejeitada: {exc}")

    reference = st.session_state.mpr_reference
    if reference:
        col1, col2, col3 = st.columns(3)
        col1.metric("Codificação", reference.encoding)
        col2.metric(
            "Fim de linha",
            {"\r\n": "CRLF", "\n": "LF", "\r": "CR"}.get(reference.newline, "?"),
        )
        col3.metric("Separador detectado", reference.decimal_separator or "não detectado")
        if reference.repeated_lines:
            with st.expander("Padrões de linha repetidos"):
                for line, count in reference.repeated_lines:
                    st.code(f"{count} × {line}")
        st.text_area(
            "Conteúdo original — somente leitura",
            reference.text,
            height=220,
            disabled=True,
        )

    if project.machine_profile:
        block = st.text_area(
            "Bloco de coordenada confirmado pelo operador",
            value=project.machine_profile.contour_block_template or "",
            placeholder="Exemplo apenas conceitual: X={{X}} Y={{Y}}",
            help=(
                "Copie a estrutura comprovada do arquivo real e troque somente os "
                "valores de coordenadas por {{X}} e {{Y}}. O bloco é repetido por ponto."
            ),
        )
        project.machine_profile.contour_block_template = block or None
    if reference:
        st.session_state.mpr_program_template = st.text_area(
            "Modelo interno do programa",
            value=st.session_state.mpr_program_template,
            height=260,
            help=(
                "Edite a cópia e substitua apenas a região de contornos por "
                "{{CONTOUR_BLOCKS}}. Também são aceitos {{PROJECT_NAME}}, {{LENGTH}}, "
                "{{WIDTH}} e {{THICKNESS}}."
            ),
        )

    processor = TemplateMPRPostProcessor(
        project.machine_profile,
        reference,
        st.session_state.mpr_program_template,
    )
    mpr_issues = processor.validate_profile()
    all_errors = has_critical_errors(geometry_issues) or has_critical_errors(mpr_issues)
    if mpr_issues:
        with st.expander("Verificações específicas do MPR", expanded=all_errors):
            render_issue_list(mpr_issues)
    if all_errors:
        st.error(
            "Não foi possível gerar um MPR confiável com este arquivo de referência. "
            "Exporte o DXF e faça a conversão no software oficial da máquina."
        )
        return
    try:
        output = processor.generate(project, project.contours)
        encoding = reference.encoding if reference else "utf-8"
        output_bytes = output.encode(encoding)
        st.download_button(
            "Baixar MPR experimental",
            data=output_bytes,
            file_name=f"{safe_filename(project.name, 'projeto')}.mpr",
            mime="text/plain",
            use_container_width=True,
        )
    except (ValueError, UnicodeEncodeError) as exc:
        st.error(f"MPR bloqueado: {exc}")


def stage_validate_export(project: CNCProject) -> None:
    st.header("8 — Validar e exportar")
    issues = validate_project(project)
    project.validation_issues = issues
    render_issue_list(issues)
    error_count = sum(
        issue.severity == ValidationSeverity.ERROR for issue in issues
    )
    warning_count = sum(
        issue.severity == ValidationSeverity.WARNING for issue in issues
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Erros", error_count)
    col2.metric("Avisos", warning_count)
    col3.metric("Contornos selecionados", len(project.selected_contours))
    dxf_tab, mpr_tab = st.tabs(["Exportar DXF", "Exportar MPR"])
    with dxf_tab:
        render_dxf_export(project, issues)
    with mpr_tab:
        render_mpr_export(project, issues)


def main() -> None:
    initialize_state()
    project: CNCProject = st.session_state.project
    st.title("Imagem para linhas vetoriais CNC")
    st.warning(SAFETY_NOTICE)
    stage = render_project_sidebar(project)
    renderers = {
        STAGES[0]: stage_upload,
        STAGES[1]: stage_calibration,
        STAGES[2]: stage_processing,
        STAGES[3]: stage_contours,
        STAGES[4]: stage_geometry,
        STAGES[5]: stage_preview,
        STAGES[6]: stage_tool,
        STAGES[7]: stage_validate_export,
    }
    renderers[stage](project)


if __name__ == "__main__":
    main()
