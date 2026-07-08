from __future__ import annotations

from pathlib import Path

from .service import KnowledgeService


def create_app(data_root: Path):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("install the 'ui' optional dependencies") from exc

    service = KnowledgeService.from_data_root(data_root)
    standard_choices = [""] + service.standards

    def search(query: str, standard: str, limit: int):
        rows = service.search(query.strip(), standard_code=standard, limit=int(limit))
        return [
            [
                row["score"],
                row["standard_code"],
                row["page"],
                row["section"],
                str(row["text"])[:320] + ("…" if len(str(row["text"])) > 320 else ""),
            ]
            for row in rows
        ]

    def path(indicator: str):
        rows = service.path(indicator.strip())
        return [
            [
                row["head"],
                row["relation"],
                row["tail"],
                row["standard_code"],
                row["page"],
                row["section"],
            ]
            for row in rows
        ]

    def missing(indicator: str):
        return service.missing(indicator.strip())

    def evidence(query: str, standard: str):
        return service.evidence(query.strip(), standard_code=standard)

    css = """
    .gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
    .action-button { width: 128px !important; min-width: 128px !important; }
    @media (max-width: 600px) {
        .gradio-container { padding-left: 12px !important; padding-right: 12px !important; }
        .tab-container {
            display: grid !important;
            grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
            overflow: visible !important;
        }
        .tab-container button {
            min-width: 0 !important;
            padding: 8px 1px !important;
            font-size: 12px !important;
        }
        .mobile-stack {
            flex-direction: column !important;
            flex-wrap: nowrap !important;
        }
        .mobile-stack > * { width: 100% !important; min-width: 0 !important; }
        .json-holder { max-width: 100% !important; overflow-x: auto !important; }
    }
    """
    with gr.Blocks(title="EcoSpec-KG", css=css) as demo:
        gr.Markdown("# EcoSpec-KG")
        with gr.Tabs():
            with gr.Tab("规范检索"):
                with gr.Row(elem_classes="mobile-stack"):
                    query = gr.Textbox(label="检索词", value="水源涵养")
                    standard = gr.Dropdown(
                        label="标准", choices=standard_choices, value=""
                    )
                    limit = gr.Slider(1, 20, value=10, step=1, label="返回条数")
                search_button = gr.Button(
                    "检索",
                    variant="primary",
                    size="sm",
                    scale=0,
                    min_width=128,
                    elem_classes="action-button",
                )
                search_output = gr.Dataframe(
                    headers=["得分", "标准", "页码", "章节", "证据文本"],
                    datatype=["number", "str", "number", "str", "str"],
                    interactive=False,
                    wrap=True,
                )
                search_button.click(
                    search,
                    inputs=[query, standard, limit],
                    outputs=search_output,
                    api_name="search",
                )
            with gr.Tab("指标依赖路径"):
                indicator = gr.Textbox(label="指标", value="水源涵养量")
                path_button = gr.Button(
                    "生成路径",
                    variant="primary",
                    size="sm",
                    scale=0,
                    min_width=128,
                    elem_classes="action-button",
                )
                path_output = gr.Dataframe(
                    headers=["头实体", "关系", "尾实体", "标准", "页码", "章节"],
                    interactive=False,
                    wrap=True,
                )
                path_button.click(
                    path, inputs=indicator, outputs=path_output, api_name="path"
                )
            with gr.Tab("缺失项检查"):
                missing_indicator = gr.Textbox(label="指标", value="水源涵养量")
                missing_button = gr.Button(
                    "检查",
                    variant="primary",
                    size="sm",
                    scale=0,
                    min_width=128,
                    elem_classes="action-button",
                )
                missing_output = gr.JSON(label="检查结果")
                missing_button.click(
                    missing,
                    inputs=missing_indicator,
                    outputs=missing_output,
                    api_name="missing",
                )
            with gr.Tab("原文证据"):
                with gr.Row(elem_classes="mobile-stack"):
                    evidence_query = gr.Textbox(label="检索词", value="水量平衡方程")
                    evidence_standard = gr.Dropdown(
                        label="标准", choices=standard_choices, value=""
                    )
                evidence_button = gr.Button(
                    "定位证据",
                    variant="primary",
                    size="sm",
                    scale=0,
                    min_width=128,
                    elem_classes="action-button",
                )
                evidence_output = gr.JSON(label="证据")
                evidence_button.click(
                    evidence,
                    inputs=[evidence_query, evidence_standard],
                    outputs=evidence_output,
                    api_name="evidence",
                )
    return demo


def launch(data_root: Path, host: str, port: int) -> None:
    app = create_app(data_root)
    app.launch(server_name=host, server_port=port, show_error=True)
