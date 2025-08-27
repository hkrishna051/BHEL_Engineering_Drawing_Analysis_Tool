# load the csv data - arc.csv at line no 1204

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging
import os
import sys
from scipy import stats
import json

# Preference - Ignore warnings
warnings.filterwarnings('ignore')

# Set up logging - info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drawing_analysis.log'),
        logging.StreamHandler()
    ]
)

# Plotting Style
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white'
})

class DrawingAnalysisSystem:
    """
    BHEL Engineering Drawing Analysis Tool
    Built during internship (Harekrishna) at BHEL for analyzing drawing review processes.
    This tool helps identify bottlenecks and inefficiencies in the drawing
    approval workflow.

    """

    def __init__(self, csv_file):
        self.data_file = csv_file
        self.raw_df = None
        self.processed_df = None
        self.results = {}
        
        # Configuration
        self.config = {
            'outlier_multiplier': 1.5,  # Standard IQR multiplier
            'max_reasonable_review_days': 365,  # Anything beyond this is likely data error
            'chart_colors': ['#1f77b4', "#2B2A29", "#3b4c3b", "#392828", '#9467bd', "#4c241c"],
            'figure_dpi': 300  # set quality for presentations
        }
        
        # separate output directory for customer analysis
        self.customer_output_dir = 'customer_analysis_results'
        
        print("Initializing Drawing Analysis System...")
        self._load_and_validate_data()

    def _load_and_validate_data(self):
        """Load the CSV file data"""
        try:
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"Cannot find file: {self.data_file}")
            
            print(f"Loading data from {self.data_file}...")
            self.raw_df = pd.read_csv(self.data_file)
            
            print(f"Loaded {len(self.raw_df):,} records successfully")
            print(f"Dataset shape: {self.raw_df.shape}")
            
            print("\nAvailable columns:")
            for i, col in enumerate(self.raw_df.columns, 1):
                print(f"  {i:2d}. {col}")
            
            missing_data = self.raw_df.isnull().sum()
            if missing_data.sum() > 0:
                print(f"\nFound missing data in {missing_data[missing_data > 0].shape[0]} columns")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def clean_and_process_data(self):
        """Clean the raw data and create derived fields"""
        print("\nStarting data cleaning and processing...")
        
        if self.raw_df is None:
            raise ValueError("No raw data available. Check data loading.")
        
        df = self.raw_df.copy()
        
        # Column mapping
        column_mapping = {
            'ID': 'record_id',
            'DRAWING_ID': 'drawing_id',
            'DRAWING_NO': 'drawing_number',
            'DRAWING_TITLE': 'title',
            'SUBMISSION_DATE': 'submitted_on',
            'REVIEWED_ON': 'reviewed_on',
            'RE_SUBMISSION_REQUIRED': 'needs_resubmission',
            'REVISION_NO': 'revision_num',
            'PROJECT': 'project_name',
            'BHEL_UNIT': 'unit',
            'PROJECT_TYPE': 'project_type',
            'TYPE_OF_DOCUMENT': 'doc_type',
            'REGION': 'region',
            'CUSTOMER': 'customer'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]
        
        df = self._fix_date_columns(df)
        df = self._calculate_review_metrics(df)
        df = self._categorize_drawings(df)
        df = self._handle_missing_data(df)
        df = self._find_outliers(df)
        
        self.processed_df = df
        print("Data processing completed!")
        return df

    def _fix_date_columns(self, df):
        """Fix date columns by determining the format from the data"""
        date_cols = ['submitted_on', 'reviewed_on']
        
        for col in date_cols:
            if col in df.columns:
                print(f"Processing {col}...")
                
                # different date formats
                date_formats = ['%d-%m-%y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']
                
                for fmt in date_formats:
                    try:
                        df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                        if not df[col].isna().all():
                            break
                    except:
                        continue
                
                if df[col].isna().all():
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                success_rate = (1 - df[col].isna().sum() / len(df)) * 100
                print(f"Successfully converted {success_rate:.1f}% dates")
        
        return df

    def _calculate_review_metrics(self, df):
        """ review-related metrics and status"""
        print("Calculating review metrics...")
        
        if 'submitted_on' in df.columns and 'reviewed_on' in df.columns:
            df['review_days'] = (df['reviewed_on'] - df['submitted_on']).dt.days
            
            def get_review_status(row):
                if pd.isna(row['submitted_on']):
                    return 'Not Submitted'
                elif pd.isna(row['reviewed_on']):
                    return 'Under Review'
                elif row['reviewed_on'] < row['submitted_on']:
                    return 'Data Error'
                else:
                    return 'Completed'
            
            df['status'] = df.apply(get_review_status, axis=1)
        
        if 'drawing_id' in df.columns:
            revision_counts = df.groupby('drawing_id').size()
            df['total_revisions'] = df['drawing_id'].map(revision_counts)
        
        return df

    def _categorize_drawings(self, df):
        """Categorize drawings based on title keywords"""
        print("Categorizing drawings...")
        
        if 'title' not in df.columns:
            df['category'] = 'Unknown'
            return df
        
        categories = {
            'Electrical Systems': [
                'electrical', 'cable', 'wiring', 'panel', 'switchgear',
                'control', 'relay', 'protection'
            ],
            'Mechanical Systems': [
                'mechanical', 'pump', 'valve', 'piping', 'turbine',
                'compressor', 'motor', 'bearing'
            ],
            'Structural': [
                'structural', 'foundation', 'arrangement', 'layout',
                'building', 'platform', 'support'
            ],
            'Instrumentation': [
                'instrument', 'monitoring', 'sensor', 'measurement',
                'gauge', 'transmitter', 'indicator'
            ],
            'Fire & Safety': [
                'fire', 'safety', 'protection', 'detection', 'alarm',
                'emergency', 'evacuation'
            ],
            'Quality & Testing': [
                'quality', 'test', 'inspection', 'procedure', 'plan',
                'specification', 'standard'
            ],
            'Transformer': [
                'transformer', 'trfr', 'bushing', 'winding', 'core',
                'tank', 'cooling'
            ],
            'Excitation System': [
                'excitation', 'static', 'inverter', 'thyristor',
                'rectifier', 'converter'
            ]
        }
        
        def categorize_single_drawing(title):
            if pd.isna(title):
                return 'Uncategorized'
            
            title_lower = str(title).lower()
            for category, keywords in categories.items():
                if any(keyword in title_lower for keyword in keywords):
                    return category
            return 'Other'
        
        df['category'] = df['title'].apply(categorize_single_drawing)
        cat_counts = df['category'].value_counts()
        print(f"Found {len(cat_counts)} categories")
        
        return df

    def _handle_missing_data(self, df):
        """Handling  missing values """
        print("Handling missing values...")
        
        if 'revision_num' in df.columns:
            df['revision_num'] = df['revision_num'].fillna(0)
        
        if 'needs_resubmission' in df.columns:
            df['needs_resubmission'] = df['needs_resubmission'].fillna('N')
        
        if 'project_type' in df.columns:
            df['project_type'] = df['project_type'].fillna('Standard')
        
        return df

    def _find_outliers(self, df):
        """Find outliers in review times using IQR method"""
        print("Detecting outliers...")
        
        if 'review_days' not in df.columns:
            df['is_outlier'] = False
            return df
        
        valid_reviews = df['review_days'].dropna()
        valid_reviews = valid_reviews[
            (valid_reviews >= 0) & 
            (valid_reviews <= self.config['max_reasonable_review_days'])
        ]
        
        if len(valid_reviews) == 0:
            df['is_outlier'] = False
            return df
        
        Q1 = valid_reviews.quantile(0.25)
        Q3 = valid_reviews.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.config['outlier_multiplier'] * IQR
        upper_bound = Q3 + self.config['outlier_multiplier'] * IQR
        
        df['is_outlier'] = (
            (df['review_days'] < lower_bound) | 
            (df['review_days'] > upper_bound)
        ).fillna(False)
        
        outlier_count = df['is_outlier'].sum()
        outlier_pct = (outlier_count / len(df)) * 100
        print(f"Found {outlier_count} outliers ({outlier_pct:.1f}%)")
        
        return df

    def analyze_data(self):
        """Performing comprehensive analysis"""
        print("\nStarting comprehensive analysis...")
        
        if self.processed_df is None:
            print("No processed data found. Running data processing first...")
            self.clean_and_process_data()
        
        drawing_analysis = self._analyze_by_drawing()
        project_analysis = self._analyze_by_project()
        unit_analysis = self._analyze_by_unit()
        customer_analysis = self._analyze_by_customer()
        overall_stats = self._calculate_overall_stats()
        
        self.results = {
            'drawing_level': drawing_analysis,
            'project_level': project_analysis,
            'unit_level': unit_analysis,
            'customer_level': customer_analysis,
            'overall': overall_stats,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print("Analysis completed!")
        return self.results

    def _analyze_by_drawing(self):
        """Analyze metrics at drawing level and calculate efficiency score"""
        print("Analyzing individual drawings...")
        df = self.processed_df
        
        if 'drawing_id' not in df.columns:
            print("No drawing ID found, skipping drawing-level analysis")
            return pd.DataFrame()
        
        drawing_stats = df.groupby('drawing_id').agg({
            'record_id': 'count',
            'review_days': ['mean', 'min', 'max', 'std'],
            'needs_resubmission': lambda x: (x == 'Y').sum(),
            'is_outlier': 'sum',
            'submitted_on': ['min', 'max'],
            'status': lambda x: (x == 'Completed').sum() / len(x) * 100
        }).round(2)
        
        drawing_stats.columns = [
            'total_records', 'avg_review_days', 'min_review_days',
            'max_review_days', 'std_review_days', 'resubmissions',
            'outlier_count', 'first_submission', 'last_submission',
            'completion_rate'
        ]
        
        # Calculate efficiency score at the drawing level
        drawing_stats['efficiency_score'] = self._calculate_efficiency_score_drawing(drawing_stats)
        
        print(f"Analyzed {len(drawing_stats)} unique drawings")
        return drawing_stats

    def _analyze_by_project(self):
        """Analyze metrics at project level"""
        print("Analyzing projects...")
        df = self.processed_df
        
        if 'project_name' not in df.columns:
            print("No project info found, skipping project analysis")
            return pd.DataFrame()
        
        project_stats = df.groupby('project_name').agg({
            'drawing_id': 'nunique',
            'review_days': 'mean',
            'status': lambda x: (x == 'Completed').sum() / len(x) * 100,
            'is_outlier': 'sum',
            'needs_resubmission': lambda x: (x == 'Y').sum()
        }).round(2)
        
        project_stats.columns = [
            'unique_drawings', 'avg_review_time', 'completion_rate',
            'outliers', 'resubmissions'
        ]
        
        print(f"Analyzed {len(project_stats)} projects")
        return project_stats

    def _analyze_by_unit(self):
        """Analyze metrics at unit level, including efficiency score"""
        print("Analyzing units...")
        df = self.processed_df
        
        if 'unit' not in df.columns:
            print("No unit info found, skipping unit analysis")
            return pd.DataFrame()
        
        unit_stats = df.groupby('unit').agg({
            'record_id': 'count',
            'drawing_id': 'nunique',
            'review_days': ['mean', 'median', 'min', 'max', 'std'],
            'needs_resubmission': lambda x: (x == 'Y').sum(),
            'is_outlier': 'sum',
            'status': lambda x: (x == 'Completed').sum() / len(x) * 100
        }).round(2)
        
        unit_stats.columns = [
            'total_records', 'unique_drawings', 'avg_review_days',
            'median_review_days', 'min_review_days', 'max_review_days',
            'std_review_days', 'resubmissions', 'outlier_count',
            'completion_rate'
        ]
        
        # Calculate efficiency score at the unit level
        unit_stats['efficiency_score'] = self._calculate_efficiency_score_unit(unit_stats)
        
        print(f"Analyzed {len(unit_stats)} units")
        return unit_stats

    def _analyze_by_customer(self):
        """Analyze metrics at customer level"""
        print("Analyzing customers...")
        df = self.processed_df
        
        if 'customer' not in df.columns:
            print("No customer info found, skipping customer analysis")
            return pd.DataFrame()
        
        customer_stats = df.groupby('customer').agg({
            'record_id': 'count',
            'drawing_id': 'nunique',
            'review_days': ['mean', 'median', 'min', 'max', 'std'],
            'needs_resubmission': lambda x: (x == 'Y').sum(),
            'is_outlier': 'sum',
            'status': lambda x: (x == 'Completed').sum() / len(x) * 100
        }).round(2)
        
        customer_stats.columns = [
            'total_records', 'unique_drawings', 'avg_review_days',
            'median_review_days', 'min_review_days', 'max_review_days',
            'std_review_days', 'resubmissions', 'outlier_count',
            'completion_rate'
        ]
        
        print(f"Analyzed {len(customer_stats)} customers")
        return customer_stats

    def _calculate_overall_stats(self):
        """Calculate overall summary statistics"""
        print("Calculating overall statistics...")
        df = self.processed_df
        
        stats = {
            'total_records': len(df),
            'unique_drawings': df['drawing_id'].nunique() if 'drawing_id' in df.columns else 0,
            'unique_projects': df['project_name'].nunique() if 'project_name' in df.columns else 0,
            'unique_units': df['unit'].nunique() if 'unit' in df.columns else 0,
            'unique_customers': df['customer'].nunique() if 'customer' in df.columns else 0,
            'avg_review_time': df['review_days'].mean(),
            'median_review_time': df['review_days'].median(),
            'completion_rate': (df['status'] == 'Completed').sum() / len(df) * 100,
            'outlier_rate': df['is_outlier'].sum() / len(df) * 100,
            'resubmission_rate': (df['needs_resubmission'] == 'Y').sum() / len(df) * 100,
            'data_completeness': self._calculate_data_completeness(df)
        }
        
        return stats

    def _calculate_efficiency_score_drawing(self, drawing_stats):
        """Calculate custom  efficiency score for drawings for time, revison and completion"""
        time_score = np.clip(100 - drawing_stats['avg_review_days'] * 1.5, 0, 100)
        revision_score = np.clip(100 - drawing_stats['total_records'] * 8, 0, 100)
        completion_score = drawing_stats['completion_rate']
        
        efficiency = (time_score * 0.4 + revision_score * 0.3 + completion_score * 0.3)
        return efficiency.round(1)

    def _calculate_efficiency_score_unit(self, unit_stats):
        """Calculate custom efficiency score for each units"""
        time_score = np.clip(100 - unit_stats['avg_review_days'] * 0.5, 0, 100)
        resubmission_score = np.clip(100 - unit_stats['resubmissions'] / unit_stats['total_records'] * 100 * 1.5, 0, 100)
        completion_score = unit_stats['completion_rate']
        
        efficiency = (time_score * 0.4 + resubmission_score * 0.3 + completion_score * 0.3)
        return efficiency.round(1)

    def _calculate_data_completeness(self, df):
        """Calculate for how complete our data is"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        return round(completeness, 1)

    def create_dashboard(self, save_dir='analysis_results'):
        """Create the main visualization dashboard"""
        print("\nCreating visualization dashboard...")
        
        if not self.results:
            print("No analysis results found. Running analysis first...")
            self.analyze_data()
        
        os.makedirs(save_dir, exist_ok=True)
        
        self._create_main_dashboard(save_dir)
        self._create_detailed_charts(save_dir)
        
        print(f"Dashboard saved to {save_dir}/")

    def _create_main_dashboard(self, save_dir):
        """ The main comprehensive dashboard and individual plot PNG's"""
        fig = plt.figure(figsize=(36, 20))
        plt.style.use('seaborn-v0_8')
        
        fig.suptitle('BHEL Engineering Drawing Analysis Dashboard',
                    fontsize=24, fontweight='bold', y=0.98, ha='center')
        
        gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3,
                             top=0.95, bottom=0.08, left=0.06, right=0.94)
        
        # Dictionary to store individual plot figures
        individual_figures = {}
        
        # 1. Review Status Overview (Pie Chart)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_status_overview(ax1)
        ax1.set_title('Review Status Distribution', fontsize=16, fontweight='bold', pad=15) #Can adjust fig sizes and internals accoring to needs and similarly for the other pngs.
        individual_figures['status_overview'] = (self._plot_status_overview, 'Review Status Distribution')
        
        # 2. Drawing Categories (Bar Chart)
        ax2 = fig.add_subplot(gs[0, 2:4])
        self._plot_category_distribution(ax2)
        ax2.set_title('Drawing Categories', fontsize=16, fontweight='bold', pad=15)
        individual_figures['category_distribution'] = (self._plot_category_distribution, 'Drawing Categories')
        
        # 3. Monthly Submission Trends (Line Chart)
        ax3 = fig.add_subplot(gs[0, 4:])
        self._plot_monthly_trends(ax3)
        ax3.set_title('Monthly Submission Trends', fontsize=16, fontweight='bold', pad=15)
        individual_figures['monthly_trends'] = (self._plot_monthly_trends, 'Monthly Submission Trends')
        
        # 4. Review Time Distribution (Histogram)
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_review_time_distribution(ax4)
        ax4.set_title('Review Time Distribution', fontsize=16, fontweight='bold', pad=15)
        individual_figures['review_time_distribution'] = (self._plot_review_time_distribution, 'Review Time Distribution')
        
        # 5. Top Drawings by Revisions (Horizontal Bar)
        ax5 = fig.add_subplot(gs[1, 2:4])
        self._plot_top_revisions(ax5)
        ax5.set_title('Top Drawings by Revisions', fontsize=16, fontweight='bold', pad=15)
        individual_figures['top_revisions'] = (self._plot_top_revisions, 'Top Drawings by Revisions')
        
        # 6. Project Performance Comparison (Grouped Bar)
        ax6 = fig.add_subplot(gs[1, 4:])
        self._plot_project_comparison(ax6)
        ax6.set_title('Project Performance Comparison', fontsize=16, fontweight='bold', pad=15)
        individual_figures['project_comparison'] = (self._plot_project_comparison, 'Project Performance Comparison')
        
        # 7. Outlier Analysis (Box Plot)
        ax7 = fig.add_subplot(gs[2, :3])
        self._plot_outlier_analysis(ax7)
        ax7.set_title('Outlier Analysis', fontsize=16, fontweight='bold', pad=15)
        individual_figures['outlier_analysis'] = (self._plot_outlier_analysis, 'Outlier Analysis')
        
        # 8. Resubmission Analysis (Bar Chart)
        ax8 = fig.add_subplot(gs[2, 3:])
        self._plot_resubmission_analysis(ax8)
        ax8.set_title('Resubmission Analysis', fontsize=16, fontweight='bold', pad=15)
        individual_figures['resubmission_analysis'] = (self._plot_resubmission_analysis, 'Resubmission Analysis')
        
        # 9. Summary Statistics Table
        ax9 = fig.add_subplot(gs[3, :])
        self._create_summary_table(ax9)
        ax9.set_title('Summary Statistics', fontsize=16, fontweight='bold', pad=15)
        individual_figures['summary_table'] = (self._create_summary_table, 'Summary Statistics')
        
        # Apply styling
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
            ax.set_facecolor('#f8f9fa')
            ax.grid(True, alpha=0.3, linestyle='--')
            for spine in ax.spines.values():
                spine.set_color('#cccccc')
                spine.set_linewidth(0.5)
            ax.tick_params(axis='both', which='major', labelsize=12,
                          colors='#333333', length=4)
            ax.xaxis.label.set_color('#333333')
            ax.yaxis.label.set_color('#333333')
            ax.xaxis.label.set_fontsize(12)
            ax.yaxis.label.set_fontsize(12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        dashboard_path = os.path.join(save_dir, 'main_dashboard.png')
        plt.savefig(dashboard_path,
                    dpi=self.config.get('figure_dpi', 300),
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    pad_inches=0.2)
        plt.close(fig)
        
        print("Main dashboard saved: main_dashboard.png")
        
        # Save individual plots 
        for plot_name, (plot_func, title) in individual_figures.items():
            figsize = (10, 8) if plot_name != 'top_revisions' else (12, 8)
            fig_individual, ax_individual = plt.subplots(figsize=figsize)
            plt.style.use('seaborn-v0_8')
            
            plot_func(ax_individual)
            ax_individual.set_title(title, fontsize=16, fontweight='bold', pad=15)
            ax_individual.set_facecolor('#f8f9fa')
            ax_individual.grid(True, alpha=0.3, linestyle='--')
            
            for spine in ax_individual.spines.values():
                spine.set_color('#cccccc')
                spine.set_linewidth(0.5)
            ax_individual.tick_params(axis='both', which='major', labelsize=12,
                                     colors='#333333', length=4)
            ax_individual.xaxis.label.set_color('#333333')
            ax_individual.yaxis.label.set_color('#333333')
            ax_individual.xaxis.label.set_fontsize(12)
            ax_individual.yaxis.label.set_fontsize(12)
            
            plt.tight_layout()
            
            individual_path = os.path.join(save_dir, f'{plot_name}.png')
            plt.savefig(individual_path,
                        dpi=self.config.get('figure_dpi', 300),
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none',
                        pad_inches=0.2)
            plt.close(fig_individual)
            
            print(f"Individual plot saved: {plot_name}.png")

    def _plot_status_overview(self, ax):
        """ plot review status overview"""
        if 'status' in self.processed_df.columns:
            status_counts = self.processed_df['status'].value_counts()
            ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
                   startangle=90, colors=self.config['chart_colors'])
            ax.axis('equal')
        else:
            ax.text(0.5, 0.5, "Status column missing", ha='center', va='center', fontsize=12)
            ax.set_visible(False)

    def _plot_category_distribution(self, ax):
        """ Plot drawing category distribution"""
        if 'category' in self.processed_df.columns:
            category_counts = self.processed_df['category'].value_counts().head(10)
            category_counts.plot(kind='bar', ax=ax, color=self.config['chart_colors'])
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, "Category column missing", ha='center', va='center', fontsize=12)
            ax.set_visible(False)

    def _plot_monthly_trends(self, ax):
        """  Plot monthly submission trends"""
        if 'submitted_on' in self.processed_df.columns:
            monthly_submissions = self.processed_df.set_index('submitted_on').resample('M').size()
            if not monthly_submissions.empty:
                monthly_submissions.plot(ax=ax, marker='o', linestyle='-', 
                                       color=self.config['chart_colors'][0])
                ax.set_xlabel('Month')
                ax.set_ylabel('Number of Submissions')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, "No submission data available for plotting", 
                       ha='center', va='center', fontsize=12)
                ax.set_visible(False)
        else:
            ax.text(0.5, 0.5, "Submitted_on column missing", 
                   ha='center', va='center', fontsize=12)
            ax.set_visible(False)

    def _plot_review_time_distribution(self, ax):
        """ Plot review time distribution"""
        if 'review_days' in self.processed_df.columns:
            valid_review_times = self.processed_df['review_days'].dropna()
            if not valid_review_times.empty:
                ax.hist(valid_review_times, bins=50, color=self.config['chart_colors'][2], 
                       edgecolor='black', alpha=0.7)
                ax.set_xlabel('Review Time (Days)')
                ax.set_ylabel('Frequency')
                ax.axvline(valid_review_times.mean(), color='red', linestyle='--', 
                          label=f'Mean: {valid_review_times.mean():.1f} days')
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No valid review time data available", 
                       ha='center', va='center', fontsize=12)
                ax.set_visible(False)
        else:
            ax.text(0.5, 0.5, "Review_days column missing", 
                   ha='center', va='center', fontsize=12)
            ax.set_visible(False)

    def _plot_top_revisions(self, ax):
        """ Plot top drawings by revisions"""
        if 'drawing_id' in self.processed_df.columns and 'total_revisions' in self.processed_df.columns:
            revision_counts = self.processed_df.drop_duplicates(subset=['drawing_id']).sort_values(
                'total_revisions', ascending=False
            ).head(10)
            
            if not revision_counts.empty:
                y_pos = np.arange(len(revision_counts))
                bars = ax.barh(y_pos, revision_counts['total_revisions'], 
                              color=self.config['chart_colors'][3])
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(revision_counts['drawing_id'])
                ax.invert_yaxis()
                ax.set_xlabel('Number of Revisions')
                
                # Add value labels to the bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                            f'{width}', va='center', fontsize=9, fontweight='bold')
            else:
                ax.text(0.5, 0.5, "No revision data available", 
                       ha='center', va='center', fontsize=12)
                ax.set_visible(False)
        else:
            ax.text(0.5, 0.5, "Required columns missing for revisions analysis", 
                   ha='center', va='center', fontsize=12)
            ax.set_visible(False)

    def _plot_project_comparison(self, ax):
        """ Plot project performance comparison"""
        if 'project_name' in self.processed_df.columns and 'review_days' in self.processed_df.columns and 'status' in self.processed_df.columns:
            project_summary = self.processed_df.groupby('project_name').agg(
                avg_review_time=('review_days', 'mean'),
                completion_rate=('status', lambda x: (x == 'Completed').sum() / len(x) * 100)
            ).round(2)
            
            if not project_summary.empty:
                project_summary[['avg_review_time', 'completion_rate']].plot(
                    kind='bar', ax=ax, secondary_y='completion_rate', 
                    color=[self.config['chart_colors'][0], self.config['chart_colors'][1]]
                )
                ax.set_xlabel('Project')
                ax.set_ylabel('Average Review Time (Days)')
                ax.right_ax.set_ylabel('Completion Rate (%)')
                ax.tick_params(axis='x', rotation=45)
                ax.legend(['Avg Review Time'], loc='upper left', fontsize=9)
                ax.right_ax.legend(['Completion Rate'], loc='upper right', fontsize=9)
            else:
                ax.text(0.5, 0.5, "No project data available for comparison", 
                       ha='center', va='center', fontsize=12)
                ax.set_visible(False)
        else:
            ax.text(0.5, 0.5, "Required columns missing for project comparison", 
                   ha='center', va='center', fontsize=12)
            ax.set_visible(False)

    def _plot_outlier_analysis(self, ax):
        """ Plot outlier analysis"""
        if 'review_days' in self.processed_df.columns and 'is_outlier' in self.processed_df.columns:
            sns.boxplot(x='is_outlier', y='review_days', data=self.processed_df, ax=ax, 
                       palette='viridis')
            ax.set_xlabel('Is Outlier')
            ax.set_ylabel('Review Time (Days)')
            ax.set_xticklabels(['Not Outlier', 'Outlier'])
            ax.set_ylim(-10, self.processed_df['review_days'].quantile(0.99) * 1.5)
        else:
            ax.text(0.5, 0.5, "Required columns missing for outlier analysis", 
                   ha='center', va='center', fontsize=12)
            ax.set_visible(False)

    def _plot_resubmission_analysis(self, ax):
        """ Plot resubmission analysis"""
        if 'needs_resubmission' in self.processed_df.columns:
            resubmission_counts = self.processed_df['needs_resubmission'].value_counts()
            if not resubmission_counts.empty:
                resubmission_counts.plot(kind='bar', ax=ax, 
                                       color=[self.config['chart_colors'][0], self.config['chart_colors'][3]])
                ax.set_xlabel('Resubmission Required')
                ax.set_ylabel('Count')
                ax.set_xticklabels(resubmission_counts.index, rotation=0)
                
                # Add count labels on top of bars
                for i, count in enumerate(resubmission_counts):
                    ax.text(i, count + max(resubmission_counts)*0.01, str(count), 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax.text(0.5, 0.5, "No resubmission data available", 
                       ha='center', va='center', fontsize=12)
                ax.set_visible(False)
        else:
            ax.text(0.5, 0.5, "Needs_resubmission column missing", 
                   ha='center', va='center', fontsize=12)
            ax.set_visible(False)

    def _create_summary_table(self, ax):
        """ Create summary statistics table"""
        if self.results.get('overall'):
            stats = self.results['overall']
            
            summary_data = {
                'Metric': [
                    'Total Records', 'Unique Drawings', 'Unique Projects',
                    'Unique Units', 'Unique Customers', 'Avg Review Time (Days)',
                    'Median Review Time (Days)', 'Completion Rate (%)',
                    'Outlier Rate (%)', 'Resubmission Rate (%)', 'Data Completeness (%)'
                ],
                'Value': [
                    f"{stats['total_records']:,}",
                    f"{stats['unique_drawings']:,}",
                    f"{stats['unique_projects']:,}",
                    f"{stats['unique_units']:,}",
                    f"{stats['unique_customers']:,}",
                    f"{stats['avg_review_time']:.1f}",
                    f"{stats['median_review_time']:.1f}",
                    f"{stats['completion_rate']:.1f}",
                    f"{stats['outlier_rate']:.1f}",
                    f"{stats['resubmission_rate']:.1f}",
                    f"{stats['data_completeness']:.1f}"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            
            # Hide axes
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_frame_on(False)
            
            # Create table
            table = ax.table(cellText=summary_df.values,
                            colLabels=summary_df.columns,
                            loc='center',
                            cellLoc='center',
                            colColours=['#f2f2f2']*len(summary_df.columns))
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            
            # Style the table
            for (i, j) in table.get_celld().keys():
                cell = table.get_celld()[(i, j)]
                if i == 0:  # Header row
                    cell.set_text_props(fontweight='bold', color='black')
                    cell.set_facecolor('#e0e0e0')
                else:
                    cell.set_facecolor('white')
                cell.set_edgecolor('#cccccc')
        else:
            ax.text(0.5, 0.5, "Summary statistics not available", 
                   ha='center', va='center', fontsize=12)
            ax.set_visible(False)

    def _create_detailed_charts(self, save_dir):
        """ HEATMAP and SCATTER PLOT"""
        print("Creating detailed analysis charts...")
        
        # Drawing efficiency distribution
        try:
            if not self.results['drawing_level'].empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                efficiency_scores = self.results['drawing_level']['efficiency_score']
                ax.hist(efficiency_scores, bins=30, alpha=0.7,
                       color=self.config['chart_colors'][0], edgecolor='black')
                ax.axvline(efficiency_scores.mean(), color='red', linestyle='--',
                          label=f'Mean: {efficiency_scores.mean():.1f}')
                ax.set_title('Drawing Efficiency Score Distribution', fontweight='bold', fontsize=16)
                ax.set_xlabel('Efficiency Score')
                ax.set_ylabel('Number of Drawings')
                ax.legend(fontsize=10)
                ax.tick_params(axis='both', which='major', labelsize=10)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'drawing_efficiency_distribution.png'),
                           dpi=self.config['figure_dpi'], bbox_inches='tight')
                plt.close()
                print("Drawing efficiency distribution chart created")
        except Exception as e:
            print(f"Error creating drawing efficiency chart: {e}")
        
        # Unit efficiency bar chart
        try:
            if 'unit_level' in self.results and not self.results['unit_level'].empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                unit_efficiency = self.results['unit_level'].sort_values('efficiency_score', ascending=False)
                
                if not unit_efficiency.empty:
                    y_pos = range(len(unit_efficiency))
                    bars = ax.barh(y_pos, unit_efficiency['efficiency_score'],
                                  color=self.config['chart_colors'][1], alpha=0.8)
                    
                    ax.set_xlabel('Efficiency Score')
                    ax.set_ylabel('Unit')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(unit_efficiency.index, fontsize=10)
                    ax.invert_yaxis()
                    ax.set_title('Unit Efficiency Scores', fontweight='bold', fontsize=16)
                    
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                               f'{width:.1f}', ha='left', va='center', fontweight='bold', fontsize=9)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'unit_efficiency_scores_bar_chart.png'),
                               dpi=self.config['figure_dpi'], bbox_inches='tight')
                    plt.close()
                    print("Unit efficiency scores bar chart created")
        except Exception as e:
            print(f"Error creating unit efficiency bar chart: {e}")

        #  SCATTER PLOT - Unit efficiency vs drawings
        try:
            if 'unit_level' in self.results and not self.results['unit_level'].empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                unit_data = self.results['unit_level']
                
                if not unit_data.empty:
                    scatter = ax.scatter(unit_data['unique_drawings'], unit_data['efficiency_score'],
                                        alpha=0.7, color=self.config['chart_colors'][2], s=100)
                    ax.set_xlabel('Number of Unique Drawings Handled')
                    ax.set_ylabel('Efficiency Score')
                    ax.set_title('Unit Efficiency vs. Number of Drawings', fontweight='bold', fontsize=16)
                    
                    #  unit names as labels
                    for i, unit in enumerate(unit_data.index):
                        ax.annotate(unit, (unit_data['unique_drawings'][i], unit_data['efficiency_score'][i]),
                                    textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'unit_efficiency_vs_drawings_scatter.png'),
                               dpi=self.config['figure_dpi'], bbox_inches='tight')
                    plt.close()
                    print("Unit efficiency vs. drawings scatter plot created")
        except Exception as e:
            print(f"Error creating unit efficiency vs. drawings scatter plot: {e}")

        #  HEATMAP - Average review time by unit and category
        try:
            if ('unit' in self.processed_df.columns and 'category' in self.processed_df.columns and 
                'review_days' in self.processed_df.columns):
                
                # Create pivot table for categorical heatmap (Average Review Time)
                pivot_table_avg_review = self.processed_df.pivot_table(values='review_days', index='unit', 
                                                                       columns='category', aggfunc='mean')
                
                # Create pivot table for count of records
                pivot_table_count = self.processed_df.pivot_table(values='record_id', index='unit', 
                                                                 columns='category', aggfunc='count').fillna(0)
                
                if not pivot_table_avg_review.empty:
                    fig, ax = plt.subplots(figsize=(14, 10))
                    sns.heatmap(pivot_table_avg_review, annot=False, cmap='YlGnBu', ax=ax,
                               cbar_kws={'label': 'Average Review Time (Days)'})
                    
                    # annotations (Average Review Time / Count)
                    for i in range(pivot_table_avg_review.shape[0]):
                        for j in range(pivot_table_avg_review.shape[1]):
                            avg_time = pivot_table_avg_review.iloc[i, j]
                            count = pivot_table_count.iloc[i, j]
                            if not pd.isna(avg_time):
                                text = f"{avg_time:.1f}\n({int(count)})"
                                ax.text(j + 0.5, i + 0.5, text,
                                        ha='center', va='center', color='black', fontsize=8)
                            elif count > 0:
                                text = f"N/A\n({int(count)})"
                                ax.text(j + 0.5, i + 0.5, text,
                                        ha='center', va='center', color='black', fontsize=8)
                    
                    ax.set_title('Average Review Time by Unit and Category (Avg Days / Count)', 
                                fontweight='bold', fontsize=16)
                    ax.set_xlabel('Drawing Category')
                    ax.set_ylabel('Unit')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'avg_review_time_heatmap_unit_category.png'),
                               dpi=self.config['figure_dpi'], bbox_inches='tight')
                    plt.close()
                    print("Average review time heatmap (unit by category) created with counts")
        except Exception as e:
            print(f"Error creating categorical heatmap: {e}")

        # Weekly timeline chart
        try:
            if 'submitted_on' in self.processed_df.columns:
                fig, ax = plt.subplots(figsize=(15, 8))
                weekly_data = self.processed_df.groupby(
                    self.processed_df['submitted_on'].dt.to_period('W')
                ).size()
                
                if not weekly_data.empty:
                    ax.plot(range(len(weekly_data)), weekly_data.values,
                           marker='o', linewidth=1.5, markersize=4)
                    ax.set_title('Weekly Submission Timeline', fontweight='bold', fontsize=16)
                    ax.set_xlabel('Week')
                    ax.set_ylabel('Submissions')
                    step = max(1, len(weekly_data) // 20)
                    ax.set_xticks(range(0, len(weekly_data), step))
                    ax.set_xticklabels([str(weekly_data.index[i])
                                      for i in range(0, len(weekly_data), step)],
                                      rotation=45, fontsize=10)
                    ax.tick_params(axis='y', which='major', labelsize=10)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'weekly_timeline.png'),
                               dpi=self.config['figure_dpi'], bbox_inches='tight')
                    plt.close()
                    print("Weekly timeline chart created")
        except Exception as e:
            print(f"Error creating timeline chart: {e}")

    def _create_customer_charts(self, save_dir):
        """ Detailed customer analysis charts"""
        print(f"Creating customer analysis charts in {self.customer_output_dir}/...")
        
        customer_save_dir = self.customer_output_dir
        os.makedirs(customer_save_dir, exist_ok=True)
        
        customer_df = self.results.get('customer_level')
        if customer_df is None or customer_df.empty:
            print("No customer level analysis data available to create charts.")
            return
        
        # Average Review Time by Customer
        try:
            if 'avg_review_days' in customer_df.columns:
                fig, ax = plt.subplots(figsize=(10, len(customer_df) * 0.5))
                sns.barplot(x=customer_df['avg_review_days'], y=customer_df.index, 
                           ax=ax, palette='viridis')
                ax.set_title('Average Review Time by Customer', fontweight='bold', fontsize=16)
                ax.set_xlabel('Average Review Time (Days)')
                ax.set_ylabel('Customer')
                plt.tight_layout()
                plt.savefig(os.path.join(customer_save_dir, 'avg_review_time_by_customer_bar_chart.png'),
                           dpi=self.config['figure_dpi'], bbox_inches='tight')
                plt.close()
                print(f"Average review time by customer bar chart created in {customer_save_dir}/")
        except Exception as e:
            print(f"Error creating average review time by customer bar chart: {e}")

    def export_results(self, save_dir='analysis_results'):
        """Export all analysis results to files"""
        print("\nExporting analysis results...")
        
        if not self.results:
            print("No results to export. Running analysis first...")
            self.analyze_data()
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Export drawing level analysis
        if not self.results['drawing_level'].empty:
            drawing_file = os.path.join(save_dir, 'drawing_level_analysis.csv')
            self.results['drawing_level'].to_csv(drawing_file)
            print("Drawing analysis exported: drawing_level_analysis.csv")
        
        # Export project level analysis
        if not self.results['project_level'].empty:
            project_file = os.path.join(save_dir, 'project_level_analysis.csv')
            self.results['project_level'].to_csv(project_file)
            print("Project analysis exported: project_level_analysis.csv")
        
        # Export unit level analysis
        if 'unit_level' in self.results and not self.results['unit_level'].empty:
            unit_file = os.path.join(save_dir, 'unit_level_analysis.csv')
            self.results['unit_level'].to_csv(unit_file)
            print("Unit analysis exported: unit_level_analysis.csv")
        
        # Export customer level analysis
        if 'customer_level' in self.results and not self.results['customer_level'].empty:
            customer_save_dir = self.customer_output_dir
            os.makedirs(customer_save_dir, exist_ok=True)
            customer_file = os.path.join(customer_save_dir, 'customer_level_analysis.csv')
            self.results['customer_level'].to_csv(customer_file)
            print(f"Customer analysis exported: {customer_save_dir}/customer_level_analysis.csv")
        
        # Export processed data
        processed_file = os.path.join(save_dir, 'processed_data.csv')
        self.processed_df.to_csv(processed_file, index=False)
        print("Processed data exported: processed_data.csv")
        
        # Create executive summary
        self._create_executive_summary(save_dir)
        
        # Export configuration
        config_file = os.path.join(save_dir, 'analysis_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        print("Configuration exported: analysis_config.json")

    def _create_executive_summary(self, save_dir):
        """Create executive summary for management"""
        stats = self.results['overall']
        
        summary_content = f"""
BHEL ENGINEERING DRAWING ANALYSIS - EXECUTIVE SUMMARY
====================================================
Analysis Date: {self.results['analysis_date']}
Prepared by: Harekrishna (2205470) under guidence of Mr. Praksh Parida


DATASET OVERVIEW:
- Total Records Processed: {stats['total_records']:,}
- Unique Drawings: {stats['unique_drawings']:,}
- Unique Projects: {stats['unique_projects']:,}
- Unique Units: {stats['unique_units']:,}
- Unique Customers: {stats['unique_customers']:,}
- Data Completeness: {stats['data_completeness']:.1f}%

KEY PERFORMANCE INDICATORS:
- Average Review Time: {stats['avg_review_time']:.1f} days
- Median Review Time: {stats['median_review_time']:.1f} days
- Overall Completion Rate: {stats['completion_rate']:.1f}%
- Process Outlier Rate: {stats['outlier_rate']:.1f}%
- Resubmission Rate: {stats['resubmission_rate']:.1f}%

CRITICAL INSIGHTS:
1. Review Process Efficiency: The current average review time of {stats['avg_review_time']:.1f} days
   {'indicates good' if stats['avg_review_time'] < 30 else 'indicates room for improvement in'} process efficiency.

2. Quality Indicators: With a {stats['resubmission_rate']:.1f}% resubmission rate,
   {'quality standards are being maintained well' if stats['resubmission_rate'] < 15 else 'there may be opportunities to improve initial submission quality'}.

3. Process Consistency: {stats['outlier_rate']:.1f}% of reviews are statistical outliers,
   {'indicating consistent process execution' if stats['outlier_rate'] < 10 else 'suggesting process variability that may need attention'}.

STRATEGIC RECOMMENDATIONS:
1. Process Optimization: Focus on categories with highest resubmission rates
2. Resource Allocation: Consider additional resources for projects with longest review times
3. Quality Improvement: Implement pre-submission checklists to reduce resubmissions
4. Monitoring: Establish regular monitoring of outlier patterns for early intervention
5. Unit-Specific Improvements: Review unit-level efficiency scores for targeted interventions
6. Customer-Specific Review: Analyze customer-level metrics to identify specific client needs

TECHNICAL NOTES:
- Analysis performed using statistical outlier detection (IQR method)
- Efficiency scores calculated using weighted combination of time, revisions, and completion metrics
- Heatmap and scatter plot analysis included for comprehensive insights
- All visualizations and detailed metrics available in accompanying files

This analysis provides actionable insights for improving the drawing review process
and optimizing project delivery timelines.
"""
        
        summary_file = os.path.join(save_dir, 'executive_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        print("Executive summary created: executive_summary.txt")

    def run_full_analysis(self, output_dir='analysis_results'):
        """Run the complete analysis pipeline"""
        print("\nStarting complete analysis pipeline...")
        print("=" * 60)
        
        try:
            self.clean_and_process_data()
            self.analyze_data()
            self.create_dashboard(output_dir)
            self.export_results(output_dir)
            self._create_customer_charts(output_dir)
            self._print_final_summary()
            
            print("\nAnalysis pipeline completed successfully!")
            print(f"All results saved to: {output_dir}/ and customer analysis in {self.customer_output_dir}/")
            
        except Exception as e:
            print(f"\nAnalysis pipeline failed: {e}")
            raise

    def _print_final_summary(self):
        """Print final summary for presentation"""
        stats = self.results['overall']
        
        print("\n" + "="*70)
        print("BHEL ENGINEERING DRAWING ANALYSIS - FINAL SUMMARY")
        print("="*70)
        print(f"Analysis Date: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
        print(f"Records Analyzed: {stats['total_records']:,}")
        print(f"Unique Drawings: {stats['unique_drawings']:,}")
        print(f"Projects Covered: {stats['unique_projects']:,}")
        print(f"Units Analyzed: {stats['unique_units']:,}")
        print(f"Customers Analyzed: {stats['unique_customers']:,}")
        print(f"Average Review Time: {stats['avg_review_time']:.1f} days")
        print(f"Process Completion Rate: {stats['completion_rate']:.1f}%")
        print(f"Outlier Detection Rate: {stats['outlier_rate']:.1f}%")
        print(f"Resubmission Rate: {stats['resubmission_rate']:.1f}%")
        print(f"Data Quality Score: {stats['data_completeness']:.1f}%")
        
        print("\nGenerated Deliverables:")
        print("  - main_dashboard.png - Executive dashboard for presentations")
        print("  - avg_review_time_heatmap_unit_category.png - HEATMAP of review times")
        print("  - unit_efficiency_vs_drawings_scatter.png - SCATTER PLOT analysis")
        print("  - drawing_level_analysis.csv - Detailed drawing metrics")
        print("  - project_level_analysis.csv - Project performance data")
        print("  - unit_level_analysis.csv - Unit performance and efficiency")
        print("  - processed_data.csv - Clean dataset for further analysis")
        print("  - executive_summary.txt - Management summary report")
        
        print("\nKey Insights for Presentation:")
        if stats['avg_review_time'] < 30:
            print("  - Review process is performing within acceptable timeframes")
        else:
            print("  - Review times may benefit from process optimization")
        
        if stats['resubmission_rate'] < 15:
            print("  - Quality standards are being maintained effectively")
        else:
            print("  - High resubmission rate indicates quality improvement opportunities")
        
        print("="*70)
        print("Ready for presentation!")

def main():
    """Main function to run the analysis"""
    try:
        print("BHEL Engineering Drawing Analysis System")
        print("=" * 50)
        
        analyzer = DrawingAnalysisSystem('arc.csv')
        analyzer.run_full_analysis()
        
        return analyzer
        
    except Exception as e:
        print(f"System failed: {e}")
        return None

if __name__ == "__main__":
    analysis_system = main()
    
    # Check output directory
    import os
    output_dir = 'analysis_results'
    if os.path.exists(output_dir):
        print(f"Contents of the '{output_dir}' directory:")
        print(os.listdir(output_dir))
    else:
        print(f"Directory '{output_dir}' does not exist.")
