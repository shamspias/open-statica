import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
import io
import json
from datetime import datetime
import zipfile
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


class ExportService:
    """Service for exporting data and results"""

    async def export_data(self, data: pd.DataFrame, format: str, **kwargs) -> bytes:
        """Export data in various formats"""
        if format == "csv":
            return self._export_csv(data, **kwargs)
        elif format == "excel":
            return self._export_excel(data, **kwargs)
        elif format == "json":
            return self._export_json(data, **kwargs)
        elif format == "parquet":
            return self._export_parquet(data, **kwargs)
        elif format == "html":
            return self._export_html(data, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def export_results(self, results: Dict[str, Any], format: str, **kwargs) -> bytes:
        """Export analysis results"""
        if format == "pdf":
            return self._export_pdf(results, **kwargs)
        elif format == "html":
            return self._export_results_html(results, **kwargs)
        elif format == "json":
            return self._export_results_json(results, **kwargs)
        elif format == "markdown":
            return self._export_markdown(results, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def export_model(self, model: Any, format: str, **kwargs) -> bytes:
        """Export trained model"""
        if format == "pickle":
            import pickle
            buffer = io.BytesIO()
            pickle.dump(model, buffer)
            return buffer.getvalue()
        elif format == "joblib":
            import joblib
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            return buffer.getvalue()
        elif format == "onnx":
            # Convert to ONNX format
            return self._export_onnx(model, **kwargs)
        else:
            raise ValueError(f"Unsupported model export format: {format}")

    def _export_csv(self, data: pd.DataFrame, **kwargs) -> bytes:
        """Export to CSV"""
        buffer = io.StringIO()
        data.to_csv(buffer, index=kwargs.get('index', False))
        return buffer.getvalue().encode('utf-8')

    def _export_excel(self, data: pd.DataFrame, **kwargs) -> bytes:
        """Export to Excel"""
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name='Data', index=kwargs.get('index', False))

            # Add metadata sheet if provided
            if 'metadata' in kwargs:
                metadata_df = pd.DataFrame(kwargs['metadata'].items(), columns=['Key', 'Value'])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

        return buffer.getvalue()

    def _export_json(self, data: pd.DataFrame, **kwargs) -> bytes:
        """Export to JSON"""
        orient = kwargs.get('orient', 'records')
        json_str = data.to_json(orient=orient, indent=2)
        return json_str.encode('utf-8')

    def _export_parquet(self, data: pd.DataFrame, **kwargs) -> bytes:
        """Export to Parquet"""
        buffer = io.BytesIO()
        data.to_parquet(buffer, index=kwargs.get('index', False))
        return buffer.getvalue()

    def _export_html(self, data: pd.DataFrame, **kwargs) -> bytes:
        """Export to HTML"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OpenStatica Data Export</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #6366f1; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #6366f1; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Data Export</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {data.to_html(index=kwargs.get('index', False), classes='data-table')}
        </body>
        </html>
        """
        return html.encode('utf-8')

    def _export_pdf(self, results: Dict[str, Any], **kwargs) -> bytes:
        """Export results to PDF"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title = Paragraph("OpenStatica Analysis Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))

        # Metadata
        metadata = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        story.append(metadata)
        story.append(Spacer(1, 12))

        # Results
        for key, value in results.items():
            section_title = Paragraph(key.replace('_', ' ').title(), styles['Heading2'])
            story.append(section_title)

            if isinstance(value, dict):
                # Create table for dictionary data
                table_data = [[k, str(v)] for k, v in value.items()]
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
            else:
                content = Paragraph(str(value), styles['Normal'])
                story.append(content)

            story.append(Spacer(1, 12))

        doc.build(story)
        return buffer.getvalue()

    def _export_results_html(self, results: Dict[str, Any], **kwargs) -> bytes:
        """Export results to HTML"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OpenStatica Analysis Results</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    background: white;
                    border-radius: 12px;
                    padding: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                }}
                h1 {{ 
                    color: #6366f1; 
                    border-bottom: 3px solid #6366f1;
                    padding-bottom: 10px;
                }}
                h2 {{ 
                    color: #4f46e5;
                    margin-top: 30px;
                }}
                .result-section {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%;
                    margin: 15px 0;
                }}
                th, td {{ 
                    border: 1px solid #e5e7eb; 
                    padding: 12px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #6366f1; 
                    color: white;
                    font-weight: 600;
                }}
                tr:nth-child(even) {{ 
                    background-color: #f3f4f6; 
                }}
                .metadata {{
                    color: #6b7280;
                    font-size: 14px;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>OpenStatica Analysis Results</h1>
                <div class="metadata">
                    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
        """

        for key, value in results.items():
            html += f"""
                <div class="result-section">
                    <h2>{key.replace('_', ' ').title()}</h2>
                    {self._format_value_html(value)}
                </div>
            """

        html += """
            </div>
        </body>
        </html>
        """
        return html.encode('utf-8')

    def _format_value_html(self, value: Any) -> str:
        """Format value for HTML output"""
        if isinstance(value, dict):
            html = "<table>"
            for k, v in value.items():
                html += f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"
            html += "</table>"
            return html
        elif isinstance(value, list):
            return "<ul>" + "".join(f"<li>{item}</li>" for item in value) + "</ul>"
        else:
            return f"<p>{value}</p>"

    def _export_results_json(self, results: Dict[str, Any], **kwargs) -> bytes:
        """Export results to JSON"""

        # Convert numpy arrays and other non-serializable types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return obj

        json_str = json.dumps(results, default=convert, indent=2)
        return json_str.encode('utf-8')

    def _export_markdown(self, results: Dict[str, Any], **kwargs) -> bytes:
        """Export results to Markdown"""
        md = f"""# OpenStatica Analysis Results

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""

        for key, value in results.items():
            md += f"## {key.replace('_', ' ').title()}\n\n"
            md += self._format_value_markdown(value)
            md += "\n---\n\n"

        return md.encode('utf-8')

    def _format_value_markdown(self, value: Any) -> str:
        """Format value for Markdown output"""
        if isinstance(value, dict):
            md = "| Key | Value |\n|-----|-------|\n"
            for k, v in value.items():
                md += f"| {k} | {v} |\n"
            return md
        elif isinstance(value, list):
            return "\n".join(f"- {item}" for item in value) + "\n"
        else:
            return f"{value}\n"

    def _export_onnx(self, model: Any, **kwargs) -> bytes:
        """Export model to ONNX format"""
        try:
            import torch
            import onnx

            # Convert model to ONNX
            dummy_input = kwargs.get('dummy_input')
            if dummy_input is None:
                raise ValueError("dummy_input required for ONNX export")

            buffer = io.BytesIO()
            torch.onnx.export(
                model,
                dummy_input,
                buffer,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )

            return buffer.getvalue()

        except ImportError:
            raise ImportError("ONNX export requires torch and onnx packages")
