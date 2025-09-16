import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class IsskinDoctorsReportGenerator:
    def __init__(self):
        # Consistent color scheme
        self.colors = {
            'isskin': '#6366f1',           # Indigo
            'doctors': '#10b981',          # Emerald
            'agreement': '#8b5cf6',        # Violet
            'primary': '#1f2937',          # Gray 800
            'secondary': '#6b7280',        # Gray 500
            'success': '#059669',          # Emerald 600
            'warning': '#d97706',          # Amber 600
            'error': '#dc2626'             # Red 600
        }
        
        # Chart configuration
        self.chart_config = {
            'font_family': 'Inter, system-ui, sans-serif',
            'background_color': 'rgba(0,0,0,0)',
            'grid_color': 'rgba(156, 163, 175, 0.2)',
            'text_color': '#374151'
        }
    
    def load_results(self, json_file: str) -> Dict:
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return {}
    
    def create_performance_overview_chart(self, results: Dict) -> str:
        methods = ['Isskin', 'Doctors']
        binary_accuracies = []
        dx_accuracies = []
        
        # Extract binary accuracies
        binary_comp = results.get('binary_classification_comparison', {}).get('summary', {})
        binary_accuracies.append(binary_comp.get('isskin_accuracy', 0) * 100)
        binary_accuracies.append(binary_comp.get('doctors_accuracy', 0) * 100)
        
        # Extract dx accuracies
        dx_comp = results.get('dx_classification_comparison', {}).get('summary', {})
        dx_accuracies.append(dx_comp.get('isskin_accuracy', 0) * 100)
        dx_accuracies.append(dx_comp.get('doctors_accuracy', 0) * 100)
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Binary Classification (Malignant/Benign)',
            x=methods,
            y=binary_accuracies,
            marker_color=self.colors['isskin'],
            text=[f'{v:.1f}%' for v in binary_accuracies],
            textposition='auto',
            opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            name='Specific Diagnosis',
            x=methods,
            y=dx_accuracies,
            marker_color=self.colors['agreement'],
            text=[f'{v:.1f}%' for v in dx_accuracies],
            textposition='auto',
            opacity=0.8
        ))
        
        fig.update_layout(
            title={
                'text': 'Accuracy Comparison: Isskin vs Doctors',
                'font': {'size': 20, 'family': self.chart_config['font_family']},
                'x': 0.5
            },
            xaxis_title='Method',
            yaxis_title='Accuracy (%)',
            barmode='group',
            template='plotly_white',
            height=400,
            font=dict(family=self.chart_config['font_family'], color=self.chart_config['text_color']),
            plot_bgcolor=self.chart_config['background_color'],
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})
    
    def create_agreement_chart(self, results: Dict) -> str:
        classifications = ['Binary Classification', 'Specific Diagnosis']
        agreement_rates = []
        kappa_values = []
        
        # Extract agreement data
        binary_comp = results.get('binary_classification_comparison', {}).get('summary', {})
        dx_comp = results.get('dx_classification_comparison', {}).get('summary', {})
        
        agreement_rates.append(binary_comp.get('agreement_rate', 0) * 100)
        agreement_rates.append(dx_comp.get('agreement_rate', 0) * 100)
        
        kappa_values.append(binary_comp.get('cohen_kappa', 0))
        kappa_values.append(dx_comp.get('cohen_kappa', 0))
        
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Agreement rates
        fig.add_trace(
            go.Bar(
                name='Agreement Rate',
                x=classifications,
                y=agreement_rates,
                marker_color=self.colors['agreement'],
                text=[f'{v:.1f}%' for v in agreement_rates],
                textposition='auto',
                opacity=0.8
            ),
            secondary_y=False,
        )
        
        # Cohen's Kappa
        fig.add_trace(
            go.Scatter(
                name="Cohen's Kappa",
                x=classifications,
                y=kappa_values,
                mode='lines+markers',
                line=dict(color=self.colors['isskin'], width=3),
                marker=dict(size=10),
                text=[f'{v:.3f}' for v in kappa_values],
                textposition='top center'
            ),
            secondary_y=True,
        )
        
        # Add interpretation lines for kappa
        fig.add_hline(y=0.2, line_dash="dash", line_color="gray", secondary_y=True)
        fig.add_hline(y=0.4, line_dash="dash", line_color="gray", secondary_y=True)
        fig.add_hline(y=0.6, line_dash="dash", line_color="gray", secondary_y=True)
        fig.add_hline(y=0.8, line_dash="dash", line_color="gray", secondary_y=True)
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Agreement Rate (%)", secondary_y=False)
        fig.update_yaxes(title_text="Cohen's Kappa", secondary_y=True, range=[-0.2, 1.0])
        
        fig.update_layout(
            title={
                'text': 'Agreement Analysis: Isskin vs Doctors',
                'font': {'size': 18, 'family': self.chart_config['font_family']},
                'x': 0.5
            },
            template='plotly_white',
            height=400,
            font=dict(family=self.chart_config['font_family'], color=self.chart_config['text_color']),
            showlegend=True
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})
    
    def create_concordance_donut_chart(self, results: Dict, classification_type: str) -> str:
        comp_key = f'{classification_type}_classification_comparison'
        if comp_key not in results:
            return "<div class='chart-placeholder'>Concordance data not available</div>"
        
        summary = results[comp_key].get('summary', {})
        agreement_rate = summary.get('agreement_rate', 0)
        disagreement_rate = 1 - agreement_rate
        
        fig = go.Figure(data=[go.Pie(
            labels=['Concordant', 'Discordant'],
            values=[agreement_rate * 100, disagreement_rate * 100],
            hole=.6,
            marker_colors=[self.colors['success'], self.colors['error']]
        )])
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=12
        )
        
        fig.update_layout(
            title={
                'text': f'{classification_type.title()} Classification<br>Concordance Rate',
                'font': {'size': 16, 'family': self.chart_config['font_family']},
                'x': 0.5
            },
            template='plotly_white',
            height=300,
            font=dict(family=self.chart_config['font_family'], color=self.chart_config['text_color']),
            showlegend=True,
            annotations=[dict(text=f'{agreement_rate*100:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})
    
    def create_accuracy_difference_chart(self, results: Dict) -> str:
        classifications = ['Binary Classification', 'Specific Diagnosis']
        differences = []
        
        # Calculate differences (Isskin - Doctors)
        binary_comp = results.get('binary_classification_comparison', {}).get('overall_comparison', {})
        dx_comp = results.get('dx_classification_comparison', {}).get('overall_comparison', {})
        
        differences.append(binary_comp.get('accuracy_difference', 0) * 100)
        differences.append(dx_comp.get('accuracy_difference', 0) * 100)
        
        # Color based on whether Isskin performs better (positive) or worse (negative)
        colors = [self.colors['success'] if d >= 0 else self.colors['error'] for d in differences]
        
        fig = go.Figure(data=[
            go.Bar(
                x=classifications,
                y=differences,
                marker_color=colors,
                text=[f'{v:+.1f}%' for v in differences],
                textposition='auto',
                opacity=0.8
            )
        ])
        
        # Add reference line at 0
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        
        fig.update_layout(
            title={
                'text': 'Accuracy Difference: Isskin vs Doctors<br><sub>(Positive = Isskin Better, Negative = Doctors Better)</sub>',
                'font': {'size': 16, 'family': self.chart_config['font_family']},
                'x': 0.5
            },
            xaxis_title='Classification Type',
            yaxis_title='Accuracy Difference (%)',
            template='plotly_white',
            height=400,
            font=dict(family=self.chart_config['font_family'], color=self.chart_config['text_color']),
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})
    
    def create_metrics_summary_table(self, results: Dict) -> str:
        table_data = []
        
        for classification_type in ['binary', 'dx']:
            comp_key = f'{classification_type}_classification_comparison'
            if comp_key not in results:
                continue
            
            summary = results[comp_key].get('summary', {})
            overall = results[comp_key].get('overall_comparison', {})
            
            # Add summary metrics
            table_data.append({
                'Classification': classification_type.title(),
                'Metric': 'Total Analyses',
                'Value': f"{summary.get('total_analyses', 0):,}"
            })
            
            table_data.append({
                'Classification': classification_type.title(),
                'Metric': 'Isskin Predictions',
                'Value': f"{summary.get('isskin_predictions', 0):,}"
            })
            
            table_data.append({
                'Classification': classification_type.title(),
                'Metric': 'Doctor Predictions', 
                'Value': f"{summary.get('doctor_predictions', 0):,}"
            })
            
            table_data.append({
                'Classification': classification_type.title(),
                'Metric': 'Comparable Cases',
                'Value': f"{summary.get('comparable_cases', 0):,}"
            })
            
            table_data.append({
                'Classification': classification_type.title(),
                'Metric': 'Isskin Accuracy',
                'Value': f"{summary.get('isskin_accuracy', 0)*100:.1f}%"
            })
            
            table_data.append({
                'Classification': classification_type.title(),
                'Metric': 'Doctors Accuracy',
                'Value': f"{summary.get('doctors_accuracy', 0)*100:.1f}%"
            })
            
            table_data.append({
                'Classification': classification_type.title(),
                'Metric': 'Agreement Rate',
                'Value': f"{summary.get('agreement_rate', 0)*100:.1f}%"
            })
            
            table_data.append({
                'Classification': classification_type.title(),
                'Metric': "Cohen's Kappa",
                'Value': f"{summary.get('cohen_kappa', 0):.3f}"
            })
            
            accuracy_diff = overall.get('accuracy_difference', 0) * 100
            table_data.append({
                'Classification': classification_type.title(),
                'Metric': 'Accuracy Difference',
                'Value': f"{accuracy_diff:+.1f}% (Isskin vs Doctors)"
            })
        
        if not table_data:
            return "<div class='table-placeholder'>Metrics data not available</div>"
        
        df = pd.DataFrame(table_data)
        
        # Create styled HTML table
        html = """
        <div class="metrics-table-container">
            <table class="metrics-table">
                <thead>
                    <tr>
        """
        
        for col in df.columns:
            html += f"<th>{col}</th>"
        
        html += """
                    </tr>
                </thead>
                <tbody>
        """
        
        for _, row in df.iterrows():
            html += '<tr>'
            for col in df.columns:
                value = row[col]
                # Add CSS classes based on content for styling
                css_class = ""
                if col == 'Value':
                    if '%' in str(value) and 'Accuracy' in row['Metric']:
                        css_class = 'accuracy-value'
                    elif '%' in str(value) and 'Agreement' in row['Metric']:
                        css_class = 'agreement-value'
                    elif 'Kappa' in row['Metric']:
                        css_class = 'kappa-value'
                    elif 'Difference' in row['Metric']:
                        if '+' in str(value):
                            css_class = 'positive-difference'
                        elif '-' in str(value):
                            css_class = 'negative-difference'
                
                html += f'<td class="{css_class}">{value}</td>'
            html += '</tr>'
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    def generate_web_report(self, results: Dict, output_file: str):
        
        # Create chart components
        performance_chart = self.create_performance_overview_chart(results)
        agreement_chart = self.create_agreement_chart(results)
        binary_donut = self.create_concordance_donut_chart(results, 'binary')
        dx_donut = self.create_concordance_donut_chart(results, 'dx')
        accuracy_diff_chart = self.create_accuracy_difference_chart(results)
        metrics_table = self.create_metrics_summary_table(results)
        
        # Calculate summary statistics
        dataset_info = results.get('dataset_info', {})
        total_images = dataset_info.get('total_images', 0)
        total_analyses = dataset_info.get('total_analyses', 0)
        successful_predictions = dataset_info.get('successful_isskin_predictions', 0)
        success_rate = dataset_info.get('isskin_success_rate', 0) * 100
        
        # Get key metrics
        binary_summary = results.get('binary_classification_comparison', {}).get('summary', {})
        dx_summary = results.get('dx_classification_comparison', {}).get('summary', {})
        
        # HTML template with modern design
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Isskin vs Doctors Comparison Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <style>
                :root {{
                    --primary-color: #6366f1;
                    --secondary-color: #8b5cf6;
                    --success-color: #10b981;
                    --warning-color: #f59e0b;
                    --error-color: #ef4444;
                    --gray-50: #f9fafb;
                    --gray-100: #f3f4f6;
                    --gray-200: #e5e7eb;
                    --gray-300: #d1d5db;
                    --gray-500: #6b7280;
                    --gray-700: #374151;
                    --gray-800: #1f2937;
                    --gray-900: #111827;
                    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                }}
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Inter', system-ui, -apple-system, sans-serif;
                    background-color: var(--gray-50);
                    color: var(--gray-800);
                    line-height: 1.6;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 0 1rem;
                }}
                
                .header {{
                    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
                    color: white;
                    padding: 3rem 0;
                    margin-bottom: 2rem;
                    box-shadow: var(--shadow-lg);
                }}
                
                .header-content {{
                    text-align: center;
                }}
                
                .header h1 {{
                    font-size: 2.5rem;
                    font-weight: 700;
                    margin-bottom: 0.5rem;
                    letter-spacing: -0.025em;
                }}
                
                .header .subtitle {{
                    font-size: 1.25rem;
                    opacity: 0.9;
                    font-weight: 400;
                }}
                
                .header .timestamp {{
                    font-size: 0.875rem;
                    opacity: 0.7;
                    margin-top: 0.5rem;
                }}
                
                .section {{
                    background: white;
                    border-radius: 0.75rem;
                    padding: 2rem;
                    margin-bottom: 2rem;
                    box-shadow: var(--shadow-md);
                    border: 1px solid var(--gray-200);
                }}
                
                .section h2 {{
                    font-size: 1.5rem;
                    font-weight: 600;
                    color: var(--gray-900);
                    margin-bottom: 1.5rem;
                    padding-bottom: 0.75rem;
                    border-bottom: 2px solid var(--primary-color);
                }}
                
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }}
                
                .stat-card {{
                    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
                    color: white;
                    padding: 1.5rem;
                    border-radius: 0.75rem;
                    text-align: center;
                    box-shadow: var(--shadow-md);
                }}
                
                .stat-value {{
                    font-size: 2.25rem;
                    font-weight: 700;
                    margin-bottom: 0.25rem;
                }}
                
                .stat-label {{
                    font-size: 0.875rem;
                    opacity: 0.9;
                    font-weight: 500;
                }}
                
                .chart-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 2rem;
                    margin-bottom: 2rem;
                }}
                
                .chart-container {{
                    background: white;
                    border-radius: 0.5rem;
                    padding: 1rem;
                    box-shadow: var(--shadow-sm);
                    border: 1px solid var(--gray-200);
                }}
                
                .full-width-chart {{
                    grid-column: 1 / -1;
                }}
                
                .metrics-table-container {{
                    overflow-x: auto;
                    border-radius: 0.5rem;
                    border: 1px solid var(--gray-200);
                    box-shadow: var(--shadow-sm);
                }}
                
                .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.875rem;
                }}
                
                .metrics-table th {{
                    background-color: var(--gray-50);
                    color: var(--gray-700);
                    font-weight: 600;
                    padding: 0.75rem 1rem;
                    text-align: left;
                    border-bottom: 1px solid var(--gray-200);
                }}
                
                .metrics-table td {{
                    padding: 0.75rem 1rem;
                    border-bottom: 1px solid var(--gray-100);
                }}
                
                .metrics-table tbody tr:hover {{
                    background-color: var(--gray-50);
                }}
                
                .accuracy-value {{
                    font-weight: 600;
                    color: var(--primary-color);
                }}
                
                .agreement-value {{
                    font-weight: 600;
                    color: var(--secondary-color);
                }}
                
                .kappa-value {{
                    font-weight: 600;
                    color: var(--gray-700);
                }}
                
                .positive-difference {{
                    font-weight: 600;
                    color: var(--success-color);
                }}
                
                .negative-difference {{
                    font-weight: 600;
                    color: var(--error-color);
                }}
                
                .info-box {{
                    background: var(--gray-50);
                    border-left: 4px solid var(--primary-color);
                    padding: 1.5rem;
                    margin: 1.5rem 0;
                    border-radius: 0 0.5rem 0.5rem 0;
                }}
                
                .info-box h4 {{
                    color: var(--gray-900);
                    font-weight: 600;
                    margin-bottom: 0.75rem;
                }}
                
                .info-box ul {{
                    margin-left: 1.25rem;
                    color: var(--gray-700);
                }}
                
                .info-box li {{
                    margin-bottom: 0.5rem;
                }}
                
                .performance-summary {{
                    background: linear-gradient(135deg, var(--success-color) 0%, var(--primary-color) 100%);
                    color: white;
                    padding: 2rem;
                    border-radius: 0.75rem;
                    margin: 2rem 0;
                    text-align: center;
                }}
                
                .performance-summary h3 {{
                    font-size: 1.5rem;
                    margin-bottom: 1rem;
                }}
                
                .performance-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 1rem;
                    margin-top: 1rem;
                }}
                
                .performance-metric {{
                    background: rgba(255, 255, 255, 0.1);
                    padding: 1rem;
                    border-radius: 0.5rem;
                }}
                
                .performance-value {{
                    font-size: 1.5rem;
                    font-weight: 700;
                }}
                
                .performance-label {{
                    font-size: 0.875rem;
                    opacity: 0.9;
                }}
                
                .footer {{
                    text-align: center;
                    padding: 2rem 0;
                    color: var(--gray-500);
                    border-top: 1px solid var(--gray-200);
                    margin-top: 3rem;
                }}
                
                .chart-placeholder,
                .table-placeholder {{
                    text-align: center;
                    color: var(--gray-500);
                    padding: 2rem;
                    background: var(--gray-50);
                    border-radius: 0.5rem;
                    border: 2px dashed var(--gray-300);
                }}
                
                @media (max-width: 768px) {{
                    .chart-grid {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .header h1 {{
                        font-size: 2rem;
                    }}
                    
                    .section {{
                        padding: 1.5rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <div class="header-content">
                        <h1>Isskin vs Doctors</h1>
                        <p class="subtitle">Comprehensive Performance Comparison Analysis</p>
                        <p class="timestamp">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}</p>
                    </div>
                </div>
            </div>
            
            <div class="container">
                <div class="section">
                    <h2>Dataset Overview</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">{total_images}</div>
                            <div class="stat-label">Total Images</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{total_analyses}</div>
                            <div class="stat-label">Clinical Analyses</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{successful_predictions}</div>
                            <div class="stat-label">Successful Isskin Predictions</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{success_rate:.1f}%</div>
                            <div class="stat-label">Isskin Success Rate</div>
                        </div>
                    </div>
                </div>
                
                <div class="performance-summary">
                    <h3>Key Performance Metrics</h3>
                    <div class="performance-grid">
                        <div class="performance-metric">
                            <div class="performance-value">{binary_summary.get('isskin_accuracy', 0)*100:.1f}%</div>
                            <div class="performance-label">Isskin Binary Accuracy</div>
                        </div>
                        <div class="performance-metric">
                            <div class="performance-value">{binary_summary.get('doctors_accuracy', 0)*100:.1f}%</div>
                            <div class="performance-label">Doctors Binary Accuracy</div>
                        </div>
                        <div class="performance-metric">
                            <div class="performance-value">{binary_summary.get('agreement_rate', 0)*100:.1f}%</div>
                            <div class="performance-label">Binary Agreement Rate</div>
                        </div>
                        <div class="performance-metric">
                            <div class="performance-value">{dx_summary.get('agreement_rate', 0)*100:.1f}%</div>
                            <div class="performance-label">Diagnosis Agreement Rate</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Performance Comparison</h2>
                    <div class="chart-container full-width-chart">
                        {performance_chart}
                    </div>
                    <div class="info-box">
                        <h4>Performance Interpretation</h4>
                        <ul>
                            <li><strong>Binary Classification:</strong> Distinguishes between malignant and benign lesions - critical for clinical triage</li>
                            <li><strong>Specific Diagnosis:</strong> Identifies the exact type of skin lesion - important for treatment planning</li>
                            <li><strong>Concordância (Agreement):</strong> Shows how often Isskin and doctors reach the same conclusion</li>
                            <li><strong>Acurácia Relativa:</strong> Compares diagnostic accuracy against ground truth diagnoses</li>
                        </ul>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Agreement Analysis</h2>
                    <div class="chart-grid">
                        <div class="chart-container full-width-chart">
                            {agreement_chart}
                        </div>
                    </div>
                    <div class="chart-grid">
                        <div class="chart-container">
                            {binary_donut}
                        </div>
                        <div class="chart-container">
                            {dx_donut}
                        </div>
                    </div>
                    <div class="info-box">
                        <h4>Cohen's Kappa Interpretation</h4>
                        <ul>
                            <li><strong>0.00-0.20:</strong> Slight agreement</li>
                            <li><strong>0.21-0.40:</strong> Fair agreement</li>
                            <li><strong>0.41-0.60:</strong> Moderate agreement</li>
                            <li><strong>0.61-0.80:</strong> Substantial agreement</li>
                            <li><strong>0.81-1.00:</strong> Almost perfect agreement</li>
                        </ul>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Relative Performance Analysis</h2>
                    <div class="chart-container full-width-chart">
                        {accuracy_diff_chart}
                    </div>
                    <div class="info-box">
                        <h4>Performance Insights</h4>
                        <ul>
                            <li><strong>Positive values:</strong> Isskin performs better than doctors in accuracy</li>
                            <li><strong>Negative values:</strong> Doctors perform better than Isskin in accuracy</li>
                            <li><strong>Clinical Significance:</strong> Small differences may not be clinically meaningful</li>
                            <li><strong>Complementary Roles:</strong> AI and doctors may have different strengths in different scenarios</li>
                        </ul>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Detailed Metrics</h2>
                    {metrics_table}
                </div>
                
                <div class="section">
                    <h2>Clinical Implications and Recommendations</h2>
                    <div class="info-box">
                        <h4>Key Findings</h4>
                        <ul>
                            <li><strong>Concordância Análise:</strong> {binary_summary.get('agreement_rate', 0)*100:.1f}% agreement in binary classification shows {'high' if binary_summary.get('agreement_rate', 0) > 0.7 else 'moderate' if binary_summary.get('agreement_rate', 0) > 0.5 else 'low'} concordance between Isskin and doctors</li>
                            <li><strong>Diagnostic Accuracy:</strong> Isskin achieves {binary_summary.get('isskin_accuracy', 0)*100:.1f}% accuracy in binary classification vs {binary_summary.get('doctors_accuracy', 0)*100:.1f}% for doctors</li>
                            <li><strong>Specific Diagnosis:</strong> Agreement rate of {dx_summary.get('agreement_rate', 0)*100:.1f}% indicates {'strong' if dx_summary.get('agreement_rate', 0) > 0.6 else 'moderate' if dx_summary.get('agreement_rate', 0) > 0.4 else 'limited'} consensus on specific diagnoses</li>
                            <li><strong>Clinical Integration:</strong> Results suggest potential for AI-assisted diagnosis in dermatology practice</li>
                        </ul>
                    </div>
                    <div class="info-box">
                        <h4>Recommendations</h4>
                        <ul>
                            <li><strong>Complementary Tool:</strong> Use Isskin as a diagnostic support tool rather than replacement for clinical expertise</li>
                            <li><strong>Quality Control:</strong> High agreement cases may indicate reliable diagnoses</li>
                            <li><strong>Learning Opportunities:</strong> Discordant cases can be valuable for continuing medical education</li>
                            <li><strong>Workflow Integration:</strong> Consider implementing Isskin in clinical workflows for diagnostic confidence</li>
                            <li><strong>Continuous Monitoring:</strong> Track performance across different lesion types and patient populations</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <div class="container">
                    <p>Report generated by Isskin Clinical Evaluation System</p>
                    <p>For technical details and methodology, refer to the project documentation</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Web report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Isskin vs Doctors comparison web report')
    parser.add_argument('--input', required=True, help='JSON file with comparison results')
    parser.add_argument('--output', default='isskin_vs_doctors_report.html', 
                       help='Output HTML file (default: isskin_vs_doctors_report.html)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run compare_isskin_dermatologists.py first to generate the data!")
        return
    
    # Create report generator
    generator = IsskinDoctorsReportGenerator()
    
    # Load results
    print(f"Loading results from: {args.input}")
    results = generator.load_results(args.input)
    
    if not results:
        print("Error: Failed to load results!")
        return
    
    # Generate report
    print("Generating web report...")
    generator.generate_web_report(results, args.output)
    
    print(f"Report generated successfully: {args.output}")
    print("Open the HTML file in your web browser to view the report!")


if __name__ == "__main__":
    main()