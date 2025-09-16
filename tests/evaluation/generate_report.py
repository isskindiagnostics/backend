import json
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64

class ReportGenerator:
    def __init__(self):
        # Configure chart style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_results(self, json_path: str) -> dict:
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def create_confusion_matrix_plot(self, cm: list, title: str) -> str:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm_array = np.array(cm)
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benigno', 'Maligno'], 
                   yticklabels=['Benigno', 'Maligno'],
                   ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predição', fontsize=12)
        ax.set_ylabel('Verdadeiro', fontsize=12)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()
    
    def create_metrics_bar_plot(self, metrics: dict, title: str) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Acurácia', 'Sensibilidade', 'Especificidade']
        values = [
            metrics.get('accuracy', 0),
            metrics.get('sensitivity', 0),
            metrics.get('specificity', 0)
        ]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.bar(metric_names, values, color=colors, alpha=0.8)
        
        # Add values to bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}\n({value*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Valor da Métrica', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()
    
    def generate_html_report(self, results: dict, output_path: str):
        # Generate charts
        cm_plot = ""
        metrics_plot = ""
        
        if 'confusion_matrix' in results.get('binary_model', {}):
            cm_plot = self.create_confusion_matrix_plot(
                results['binary_model']['confusion_matrix'],
                'Matriz de Confusão - Modelo Binary'
            )
        
        if 'accuracy' in results.get('binary_model', {}):
            metrics_plot = self.create_metrics_bar_plot(
                results['binary_model'],
                'Métricas do Modelo Binary'
            )
        
        # HTML Template
        html_template = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Avaliação - Isskin</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 1.1rem;
        }}
        
        .timestamp {{
            background: #34495e;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            display: inline-block;
            margin-top: 15px;
            font-size: 0.9rem;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border-left: 5px solid #3498db;
        }}
        
        .card h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8rem;
            font-weight: 400;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}
        
        .metric-box {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
        }}
        
        .metric-box.sensitivity {{
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            box-shadow: 0 5px 15px rgba(231, 76, 60, 0.3);
        }}
        
        .metric-box.specificity {{
            background: linear-gradient(135deg, #2ecc71, #27ae60);
            box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        
        .metric-label {{
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 300;
        }}
        
        .chart-container {{
            text-align: center;
            margin: 25px 0;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .info-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        
        .info-label {{
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .info-value {{
            font-size: 1.1rem;
            color: #34495e;
        }}
        
        .dx-results {{
            border-left-color: #9b59b6;
        }}
        
        .error {{
            background: #ffe6e6;
            border-left-color: #e74c3c;
            color: #c0392b;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Relatório de Avaliação Isskin</h1>
            <p class="subtitle">Análise de Performance dos Modelos de Detecção de Câncer de Pele</p>
            <div class="timestamp">
                📅 Gerado em: {datetime.fromisoformat(results['timestamp']).strftime('%d/%m/%Y às %H:%M:%S')}
            </div>
        </div>

        <!-- Informações do Dataset -->
        <div class="card">
            <h2>📊 Informações do Dataset</h2>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Total de Imagens</div>
                    <div class="info-value">{results['dataset_info']['total_images']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Predições Bem-sucedidas</div>
                    <div class="info-value">{results['dataset_info']['successful_predictions']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Predições Falharam</div>
                    <div class="info-value">{results['dataset_info']['failed_predictions']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Taxa de Sucesso</div>
                    <div class="info-value">{(results['dataset_info']['successful_predictions'] / results['dataset_info']['total_images'] * 100):.1f}%</div>
                </div>
            </div>
        </div>

        <!-- Modelo Binary -->
        <div class="card">
            <h2>🔍 Modelo Binary (Maligno vs Benigno)</h2>
            
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-value">{results['binary_model']['accuracy']:.3f}</div>
                    <div class="metric-label">Acurácia ({results['binary_model']['accuracy']*100:.1f}%)</div>
                </div>
                <div class="metric-box sensitivity">
                    <div class="metric-value">{results['binary_model']['sensitivity']:.3f}</div>
                    <div class="metric-label">Sensibilidade ({results['binary_model']['sensitivity']*100:.1f}%)</div>
                </div>
                <div class="metric-box specificity">
                    <div class="metric-value">{results['binary_model']['specificity']:.3f}</div>
                    <div class="metric-label">Especificidade ({results['binary_model']['specificity']*100:.1f}%)</div>
                </div>
            </div>

            {f'<div class="chart-container"><img src="data:image/png;base64,{metrics_plot}" alt="Gráfico de Métricas"></div>' if metrics_plot else ''}
            
            {f'<div class="chart-container"><img src="data:image/png;base64,{cm_plot}" alt="Matriz de Confusão"></div>' if cm_plot else ''}

            <div style="margin-top: 20px;">
                <h3 style="color: #2c3e50; margin-bottom: 10px;">📝 Interpretação das Métricas:</h3>
                <ul style="color: #34495e; line-height: 1.8;">
                    <li><strong>Acurácia:</strong> Percentual total de acertos (casos benignos e malignos classificados corretamente)</li>
                    <li><strong>Sensibilidade:</strong> Capacidade do modelo de identificar corretamente casos malignos (evitar falsos negativos)</li>
                    <li><strong>Especificidade:</strong> Capacidade do modelo de identificar corretamente casos benignos (evitar falsos positivos)</li>
                </ul>
            </div>
        </div>

        <!-- Modelo DX -->
        <div class="card dx-results">
            <h2>🎯 Modelo DX (Diagnóstico Específico)</h2>
            
            {self._generate_dx_section(results.get('dx_model', {}))}
        </div>

        <div class="footer">
            <p>Relatório gerado automaticamente pelo sistema de avaliação Isskin</p>
            <p>Para mais informações, consulte a documentação técnica</p>
        </div>
    </div>
</body>
</html>"""

        # Save file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"📄 Relatório HTML gerado: {output_path}")
    
    def _generate_dx_section(self, dx_results: dict) -> str:
        if 'error' in dx_results:
            return f'<div class="error">❌ {dx_results["error"]}</div>'
        
        if not dx_results:
            return '<div class="error">❌ Nenhum resultado disponível para o modelo DX</div>'
        
        accuracy = dx_results.get('accuracy', 0)
        total_samples = dx_results.get('total_samples', 0)
        valid_predictions = dx_results.get('valid_predictions', 0)
        
        return f"""
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Acurácia</div>
                    <div class="info-value">{accuracy:.3f} ({accuracy*100:.1f}%)</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Amostras Totais</div>
                    <div class="info-value">{total_samples}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Predições Válidas</div>
                    <div class="info-value">{valid_predictions}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Taxa de Cobertura</div>
                    <div class="info-value">{(valid_predictions/total_samples*100):.1f}%</div>
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <h3 style="color: #2c3e50; margin-bottom: 10px;">🏷️ Classes Diagnósticas:</h3>
                <p style="color: #34495e; line-height: 1.8;">
                    O modelo DX classifica entre diferentes tipos de lesões: 
                    <strong>scc</strong> (carcinoma espinocelular), 
                    <strong>bcc</strong> (carcinoma basocelular), 
                    <strong>nevus</strong> (nevo melanocítico), 
                    <strong>seborrheic_keratosis</strong> (queratose seborreica), 
                    <strong>ak</strong> (queratose actínica), 
                    <strong>melanoma</strong>, e 
                    <strong>vasc</strong> (lesão vascular).
                </p>
            </div>
        """


def main():
    parser = argparse.ArgumentParser(description='Gera relatório HTML dos resultados de avaliação')
    parser.add_argument('json_file', help='Arquivo JSON com os resultados da avaliação')
    parser.add_argument('--output', help='Nome do arquivo HTML de saída')
    
    args = parser.parse_args()
    
    # Define output file name
    if args.output:
        output_file = args.output
    else:
        # Use same name as JSON but with .html extension
        json_path = Path(args.json_file)
        output_file = json_path.with_suffix('.html')
    
    # Generate report
    generator = ReportGenerator()
    results = generator.load_results(args.json_file)
    generator.generate_html_report(results, output_file)


if __name__ == "__main__":
    main()