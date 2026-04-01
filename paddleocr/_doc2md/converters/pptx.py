from pathlib import Path

from ..base import BaseConverter, ConvertResult
from ..exceptions import ConversionError
from ..registry import default_registry


@default_registry.register
class PptxConverter(BaseConverter):
    supported_extensions = ["pptx"]
    supported_mimetypes = [
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ]

    def convert_file(self, file_path: Path, **kwargs) -> ConvertResult:
        try:
            from pptx import Presentation
            from pptx.shapes.picture import Picture  # noqa: F401
        except ImportError:
            raise ConversionError(
                "PPTX conversion requires python-pptx: pip install paddleocr[doc2md]"
            )

        prs = Presentation(str(file_path))
        slides_md = []
        images: dict = {}
        image_counter = [0]
        slide_width = prs.slide_width

        for slide_num, slide in enumerate(prs.slides, 1):
            slide_parts = []

            # Extract slide title
            title_text = ""
            if slide.shapes.title and slide.shapes.title.text.strip():
                title_text = slide.shapes.title.text.strip()
            slide_parts.append(
                f"## Slide {slide_num}" + (f": {title_text}" if title_text else "")
            )

            # Process all shapes
            for shape in slide.shapes:
                # Skip the title shape (already handled above)
                if shape == slide.shapes.title:
                    continue
                self._process_shape(
                    shape, slide_parts, images, image_counter, slide_width, slide.part
                )

            # Speaker notes
            if slide.has_notes_slide:
                notes_text = slide.notes_slide.notes_text_frame.text.strip()
                if notes_text:
                    slide_parts.append(f"\n> **Notes**: {notes_text}")

            # Group parts by content type and separate groups with blank lines
            # to prevent HTML blocks from consuming adjacent list items
            def _classify(part: str) -> str:
                s = part.lstrip()
                if s.startswith("##"):
                    return "heading"
                if s.startswith("<"):
                    return "html"
                if s.startswith("- ") or s.lstrip().startswith("- "):
                    return "list"
                if s.startswith(">"):
                    return "blockquote"
                return "other"

            groups: list[list[str]] = []
            for part in slide_parts:
                kind = _classify(part)
                if groups and _classify(groups[-1][0]) == kind:
                    groups[-1].append(part)
                else:
                    groups.append([part])

            slides_md.append("\n\n".join("\n".join(g) for g in groups))

        md_text = "\n\n---\n\n".join(slides_md)

        return ConvertResult(
            markdown=md_text,
            images=images,
            metadata={
                "format": "PPTX",
                "slide_count": len(prs.slides),
            },
        )

    def _process_shape(
        self, shape, slide_parts, images, image_counter, slide_width, slide_part
    ):
        """Recursively process a shape: Picture, GroupShape, Chart, Table, or TextFrame."""
        from pptx.shapes.picture import Picture
        from pptx.util import Emu  # noqa: F401

        try:
            from pptx.enum.shapes import MSO_SHAPE_TYPE
        except ImportError:
            MSO_SHAPE_TYPE = None

        # 1. Picture
        if isinstance(shape, Picture):
            try:
                img = shape.image
                image_counter[0] += 1
                filename = f"image{image_counter[0]}.{img.ext}"
                rel_path = f"images/{filename}"
                images[rel_path] = img.blob
                if shape.width and slide_width:
                    pct = min(round(shape.width / slide_width * 100), 100)
                    slide_parts.append(f'<img src="images/{filename}" width="{pct}%">')
                else:
                    slide_parts.append(f'<img src="images/{filename}">')
            except (ValueError, AttributeError):
                pass
            return

        # 2. GroupShape - recurse into child shapes
        if MSO_SHAPE_TYPE and shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            try:
                for child in shape.shapes:
                    self._process_shape(
                        child,
                        slide_parts,
                        images,
                        image_counter,
                        slide_width,
                        slide_part,
                    )
            except AttributeError:
                pass
            return

        # 3. Chart
        if shape.has_chart:
            slide_parts.append(self._chart_to_md(shape.chart))
            return

        # 4. Table
        if shape.has_table:
            slide_parts.append(
                self._table_to_html(shape.table, slide_part, image_counter, images)
            )
            return

        # 5. TextFrame
        if shape.has_text_frame:
            for paragraph in shape.text_frame.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                level = paragraph.level
                indent = "  " * level
                slide_parts.append(f"{indent}- {text}")

    def _chart_to_md(self, chart) -> str:
        """Extract chart data as a Markdown table."""
        _CHART_TYPE_NAMES = {
            1: "Bar Chart",
            2: "Column Chart",
            3: "Line Chart",
            4: "Pie Chart",
            5: "Area Chart",
            51: "Doughnut Chart",
            72: "Scatter Chart",
            97: "Radar Chart",
        }
        try:
            chart_type_val = chart.chart_type.value if chart.chart_type else 0
            chart_type_name = _CHART_TYPE_NAMES.get(chart_type_val, "Chart")
        except Exception:
            chart_type_name = "Chart"

        try:
            plot = chart.plots[0]
            categories = list(plot.categories) if plot.categories else []
            series_list = list(plot.series)

            if not series_list:
                return f"[{chart_type_name}]"

            lines = [f"**{chart_type_name}**", ""]

            if categories:
                header = "| |" + "".join(f" {c} |" for c in categories)
                sep = "|---|" + "---|" * len(categories)
            else:
                max_len = max((len(list(s.values)) for s in series_list), default=0)
                header = "| |" + "".join(f" Item{i+1} |" for i in range(max_len))
                sep = "|---|" + "---|" * max_len

            lines.append(header)
            lines.append(sep)

            for idx, series in enumerate(series_list):
                try:
                    name = series.tx.text if series.tx else f"Series{idx+1}"
                except Exception:
                    name = f"Series{idx+1}"
                try:
                    values = [
                        str(round(v, 4)) if v is not None else "" for v in series.values
                    ]
                except Exception:
                    values = []
                row = f"| {name} |" + "".join(f" {v} |" for v in values)
                lines.append(row)

            return "\n".join(lines)
        except Exception:
            return f"[{chart_type_name}]"

    @staticmethod
    def _table_to_html(
        table, slide_part, image_counter_list: list, images: dict
    ) -> str:
        """Convert a PPTX table to an HTML table, handling merged cells and cell background images."""
        _BLIP_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
        _REL_NS = (
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
        )

        visited: set[tuple[int, int]] = set()
        html_parts = ["<table>"]

        for i, row in enumerate(table.rows):
            html_parts.append("<tr>")
            for j, cell in enumerate(row.cells):
                if (i, j) in visited:
                    continue
                tag = "th" if i == 0 else "td"
                attrs = ""
                if cell.is_merge_origin:
                    rs = cell.span_height
                    cs = cell.span_width
                    if cs > 1:
                        attrs += f' colspan="{cs}"'
                    if rs > 1:
                        attrs += f' rowspan="{rs}"'
                    for di in range(rs):
                        for dj in range(cs):
                            if (di, dj) != (0, 0):
                                visited.add((i + di, j + dj))

                content_parts = []

                # Extract cell background blip images
                blips = cell._tc.findall(f".//{_BLIP_NS}blip")
                for blip in blips:
                    r_embed = blip.get(f"{_REL_NS}embed")
                    if r_embed:
                        try:
                            image_part = slide_part.related_parts[r_embed]
                            image_counter_list[0] += 1
                            ext = image_part.content_type.split("/")[-1]
                            filename = f"image{image_counter_list[0]}.{ext}"
                            rel_path = f"images/{filename}"
                            images[rel_path] = image_part.blob
                            content_parts.append(
                                f'<img src="images/{filename}" width="100%">'
                            )
                        except (KeyError, AttributeError):
                            pass

                text = cell.text.strip().replace("\n", "<br>")
                if text:
                    content_parts.append(text)
                cell_html = "<br>".join(content_parts) if content_parts else ""

                html_parts.append(f"<{tag}{attrs}>{cell_html}</{tag}>")
            html_parts.append("</tr>")

        html_parts.append("</table>")
        return "\n".join(html_parts)
