import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from io import BytesIO

csv_file = 'data/simulation_data.csv'
df = pd.read_csv(csv_file)

def create_plots():
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df, kde=True)
    plt.title('Histogram and Kernel Density Estimate')
    plt.tight_layout()
    plt.savefig('histogram_kde.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.pairplot(df)
    plt.suptitle('Pairwise Distribution Plot', y=1.02)
    plt.tight_layout()
    plt.savefig('pairplot.png')
    plt.close()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column], kde=True, stat='density')
            plt.title(f'{column} Z-Distribution')
            plt.xlabel(f'{column}')
            plt.ylabel('Density')
            plt.tight_layout()
            plt.savefig(f'z_distribution_{column}.png')
            plt.close()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[column])
            plt.title(f'Boxplot of {column}')
            plt.xlabel(f'{column}')
            plt.tight_layout()
            plt.savefig(f'boxplot_{column}.png')
            plt.close()

create_plots()

def create_pdf(pdf_path):
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    normal = styles['Normal']
    title = styles['Title']
    heading = styles['Heading2']
    
    content = []

    content.append(Paragraph("Statistical Report", title))

    content.append(Paragraph("Data Summary and Detailed Analysis:", heading))

    summary = df.describe()
    summary_text = summary.to_string()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            min_val = df[column].min()
            max_val = df[column].max()
            mean_val = df[column].mean()
            median_val = df[column].median()
            std_dev = df[column].std()
            percentile_25 = df[column].quantile(0.25)
            percentile_75 = df[column].quantile(0.75)
            count = df[column].count()

            analysis_text = (
                f"<b>{column}</b>:<br/>"
                f"Range: {min_val} , {max_val}<br/>"
                f"Mean: {mean_val:.2f}<br/>"
                f"Median: {median_val:.2f}<br/>"
                f"Standard Deviation: {std_dev:.2f}<br/>"
                f"25th Percentile: {percentile_25:.2f}<br/>"
                f"75th Percentile: {percentile_75:.2f}<br/>"
                f"Data Count: {count}<br/>"
                f"Approximately 50% of the data falls within the range of {percentile_25} to {percentile_75}.<br/><br/>"
            )
            content.append(Paragraph(analysis_text, normal))

    content.append(Paragraph("Analysis of the Charts:", heading))
    content.append(Paragraph("The charts below illustrate various statistical aspects of the dataset:<br/>"
                             "- **Correlation Heatmap**: Visualizes the correlation coefficients between pairs of variables, helping to identify which variables have strong linear relationships.<br/>"
                             "- **Histogram and Kernel Density Estimate**: Provides insights into the distribution of data values and their densities, helping to understand the frequency of data points within different ranges.<br/>"
                             "- **Pairwise Distribution Plot**: Shows relationships and distributions between all pairs of variables, useful for detecting patterns and correlations across the dataset.<br/>"
                             "- **Z-Distribution Plots**: Illustrates the distribution of each variable, normalized to standard units. Useful for understanding the standard deviations and overall distribution patterns.<br/>"
                             "- **Boxplots**: Displays the distribution of data and identifies outliers for each variable.<br/><br/>", normal))

    content.append(Paragraph("Correlation Heatmap:", heading))
    content.append(Image('correlation_heatmap.png', width=6*inch, height=3*inch))

    content.append(Paragraph("Histogram and Kernel Density Estimate:", heading))
    content.append(Image('histogram_kde.png', width=6*inch, height=3*inch))

    content.append(Paragraph("Pairwise Distribution Plot:", heading))
    content.append(Image('pairplot.png', width=6*inch, height=4*inch))

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            content.append(Paragraph(f"Z-Distribution for {column}:", heading))
            content.append(Image(f'z_distribution_{column}.png', width=6*inch, height=3*inch))
            content.append(Paragraph(f"Boxplot for {column}:", heading))
            content.append(Image(f'boxplot_{column}.png', width=6*inch, height=3*inch))

    doc.build(content)

pdf_output = 'statistical_report.pdf'
create_pdf(pdf_output)

print(f"PDF report has been saved as '{pdf_output}'.")
