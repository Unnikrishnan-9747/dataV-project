from dagster import job, op, DagsterInstance, execute_job, reconstructable, Output, Out
import pandas as pd
import pymongo
from pymongo import MongoClient
import psycopg2
from psycopg2.extras import execute_batch
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.io as pio
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from fpdf import FPDF
import os
import sys
from pathlib import Path
import logging
import traceback
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns

# Configure Plotly to try Kaleido but fallback to matplotlib
pio.kaleido.scope.mathjax = None  # Disable MathJax which can cause issues

# --- Helper Functions ---

def save_plotly_figure(fig, filepath, context=None):
    """Save plotly figure with fallback to matplotlib"""
    try:
        # First try Plotly's native export
        fig.write_image(filepath, engine="kaleido", timeout=10)
        if context:
            context.log.info(f"Saved Plotly figure to {filepath}")
        return True
    except Exception as e:
        if context:
            context.log.warning(f"Plotly export failed, trying matplotlib: {str(e)}")
        
        # Fallback to matplotlib
        try:
            fig_mat = plt.figure()
            ax = fig_mat.add_subplot(111)
            
            # Convert Plotly figure to matplotlib (simple version)
            if hasattr(fig, 'data'):
                for trace in fig.data:
                    if trace.type == 'scatter':
                        ax.plot(trace.x, trace.y, 'o', label=trace.name)
                    elif trace.type == 'bar':
                        ax.bar(trace.x, trace.y, label=trace.name)
            
            ax.set_title(fig.layout.title.text if hasattr(fig.layout, 'title') else "")
            ax.legend()
            
            fig_mat.savefig(filepath)
            plt.close(fig_mat)
            if context:
                context.log.info(f"Saved matplotlib fallback to {filepath}")
            return True
        except Exception as mat_error:
            if context:
                context.log.error(f"Both Plotly and matplotlib failed: {str(mat_error)}")
            return False

# --- Data Extraction and MongoDB Storage ---

@op(out={"status": Out(), "d1_count": Out(), "d2_count": Out()})
def extract_and_store_csv_data(context):
    """Extract CSV data and store in MongoDB"""
    try:
        context.log.info("Starting CSV extraction...")
        
        # Process d1.csv (Mental Health Survey)
        d1_path = Path('employee_survey.csv')
        if not d1_path.exists():
            raise FileNotFoundError(f"Input file not found: {d1_path}")
        
        df_d1 = pd.read_csv(d1_path)
        context.log.info(f"Loaded {len(df_d1)} records from employee_survey.csv")
        
        # Process d2.csv (Employee Data)
        d2_path = Path('employee_mentalhealth1.csv')
        if not d2_path.exists():
            raise FileNotFoundError(f"Input file not found: {d2_path}")
        
        df_d2 = pd.read_csv(d2_path)
        context.log.info(f"Loaded {len(df_d2)} records from employee_mentalhealth1.csv")
        
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        try:
            client.server_info()
        except pymongo.errors.ServerSelectionTimeoutError:
            raise ConnectionError("Could not connect to MongoDB server")
            
        db = client["workforce_data"]
        
        # Store d1 data
        collection_d1 = db["mental_health_survey"]
        collection_d1.drop()
        result_d1 = collection_d1.insert_many(df_d1.to_dict('records'))
        context.log.info(f"Inserted {len(result_d1.inserted_ids)} documents to mental_health_survey")
        
        # Store d2 data
        collection_d2 = db["employee_data"]
        collection_d2.drop()
        result_d2 = collection_d2.insert_many(df_d2.to_dict('records'))
        context.log.info(f"Inserted {len(result_d2.inserted_ids)} documents to employee_data")
        
        client.close()
        
        yield Output("CSV data stored in MongoDB", output_name="status")
        yield Output(len(df_d1), output_name="d1_count")
        yield Output(len(df_d2), output_name="d2_count")
    except Exception as e:
        context.log.error(f"Error in extract_and_store_csv_data: {str(e)}")
        context.log.error(traceback.format_exc())
        raise

@op(out={"status": Out(), "record_count": Out()})
def extract_and_store_json(context):
    """Extract JSON data and store in MongoDB"""
    try:
        context.log.info("Starting JSON extraction...")
        
        json_path = Path('employee_salary.json')
        if not json_path.exists():
            raise FileNotFoundError(f"Input file not found: {json_path}")
        
        with open(json_path) as f:
            json_data = json.load(f)
        
        context.log.info(f"Loaded {len(json_data)} records from JSON file")
        
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        try:
            client.server_info()
        except pymongo.errors.ServerSelectionTimeoutError:
            raise ConnectionError("Could not connect to MongoDB server")
            
        db = client["workforce_data"]
        collection = db["raw_salaries"]
        
        collection.drop()
        result = collection.insert_many(json_data)
        context.log.info(f"Inserted {len(result.inserted_ids)} documents")
        
        client.close()
        
        yield Output("JSON data stored in MongoDB", output_name="status")
        yield Output(len(json_data), output_name="record_count")
    except Exception as e:
        context.log.error(f"Error in extract_and_store_json: {str(e)}")
        context.log.error(traceback.format_exc())
        raise

# --- ETL Processing ---

@op
def extract_from_mongodb(context, csv_status, d1_count, d2_count, json_status, json_count):
    """Extract data from MongoDB for processing"""
    try:
        context.log.info("Extracting data from MongoDB...")
        context.log.info(f"CSV status: {csv_status}, JSON status: {json_status}")
        context.log.info(f"Record counts - d1: {d1_count}, d2: {d2_count}, json: {json_count}")
        
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        try:
            client.server_info()
        except pymongo.errors.ServerSelectionTimeoutError:
            raise ConnectionError("Could not connect to MongoDB server")
            
        db = client["workforce_data"]
        
        # Extract all three datasets
        # 1. Salary data
        salary_collection = db["raw_salaries"]
        salary_df = pd.DataFrame(list(salary_collection.find({})))
        
        # 2. Mental health survey
        mental_health_collection = db["mental_health_survey"]
        mental_health_df = pd.DataFrame(list(mental_health_collection.find({})))
        
        # 3. Employee data
        employee_collection = db["employee_data"]
        employee_df = pd.DataFrame(list(employee_collection.find({})))
        
        client.close()
        
        return {
            "salary_data": salary_df,
            "mental_health_data": mental_health_df,
            "employee_data": employee_df,
            "status": "All data extracted from MongoDB"
        }
    except Exception as e:
        context.log.error(f"Error in extract_from_mongodb: {str(e)}")
        context.log.error(traceback.format_exc())
        raise

@op
def transform_data(context, data_dict):
    """Transform semi-structured data into structured format"""
    salary_df = data_dict["salary_data"]
    mental_health_df = data_dict["mental_health_data"]
    employee_df = data_dict["employee_data"]
    
    try:
        # Transform salary data
        if '_id' in salary_df.columns:
            salary_df = salary_df.drop('_id', axis=1)
        
        numeric_cols = ['salary', 'salary_in_usd', 'remote_ratio']
        salary_df[numeric_cols] = salary_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        salary_df['job_title'] = salary_df['job_title'].str.strip()
        salary_df.loc[salary_df['job_title'].str.contains('Data Scien', case=False), 'job_title'] = 'Data Scientist'
        salary_df.loc[salary_df['job_title'].str.contains('Machine Learn', case=False), 'job_title'] = 'Machine Learning Engineer'
        
        exp_mapping = {'EN': 'Entry-level', 'MI': 'Mid-level', 'SE': 'Senior', 'EX': 'Executive'}
        salary_df['experience_level_category'] = salary_df['experience_level'].map(exp_mapping)
        
        size_mapping = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
        salary_df['company_size_category'] = salary_df['company_size'].map(size_mapping)
        
        salary_df['remote_category'] = pd.cut(salary_df['remote_ratio'], 
                                     bins=[-1, 0, 50, 100], 
                                     labels=['Onsite', 'Hybrid', 'Remote'])
        
        scaler = MinMaxScaler()
        salary_df['salary_normalized'] = scaler.fit_transform(salary_df[['salary_in_usd']])
        
        # Transform mental health data
        if '_id' in mental_health_df.columns:
            mental_health_df = mental_health_df.drop('_id', axis=1)
        
        # Clean up gender column
        mental_health_df['Gender'] = mental_health_df['Gender'].str.lower()
        mental_health_df['Gender'] = mental_health_df['Gender'].replace({
            'm': 'male',
            'f': 'female',
            'male-ish': 'male',
            'maile': 'male',
            'trans-female': 'trans female',
            'something kinda male?': 'other',
            'cis female': 'female',
            'cis male': 'male'
        })
        
        # Transform employee data
        if '_id' in employee_df.columns:
            employee_df = employee_df.drop('_id', axis=1)
        
        # Clean up mental health conditions
        employee_df['Mental_Health_Condition'] = employee_df['Mental_Health_Condition'].replace({
            'None': 'No condition',
            'Burnout': 'Burnout'
        })
        
        # Create age groups
        employee_df['age_group'] = pd.cut(employee_df['Age'],
                                         bins=[0, 25, 35, 45, 55, 100],
                                         labels=['<25', '25-34', '35-44', '45-54', '55+'])
        
        context.log.info(f"Transformed data shapes - Salary: {salary_df.shape}, Mental Health: {mental_health_df.shape}, Employee: {employee_df.shape}")
        return {
            "transformed_salary": salary_df,
            "transformed_mental_health": mental_health_df,
            "transformed_employee": employee_df,
            "status": "Data transformation complete"
        }
    except Exception as e:
        context.log.error(f"Transformation error: {str(e)}")
        raise

@op
def load_to_postgres(context, transformed_data):
    """Load structured data into PostgreSQL"""
    salary_df = transformed_data["transformed_salary"]
    mental_health_df = transformed_data["transformed_mental_health"]
    employee_df = transformed_data["transformed_employee"]
    
    conn = None
    
    try:
        conn = psycopg2.connect(
            database="postgres", 
            user="postgres", 
            password="123", 
            host="localhost", 
            port="5432",
            connect_timeout=5
        )
        cur = conn.cursor()
        
        # Create tables for each dataset
        # 1. Salary data
        cur.execute("""
        DROP TABLE IF EXISTS employee_salaries;
        CREATE TABLE employee_salaries (
            work_year BIGINT,
            experience_level VARCHAR(50),
            employment_type VARCHAR(20),
            job_title VARCHAR(100),
            salary NUMERIC(12,2),
            salary_currency VARCHAR(10),
            salary_in_usd NUMERIC(12,2),
            employee_residence VARCHAR(50),
            remote_ratio BIGINT,
            company_location VARCHAR(50),
            company_size VARCHAR(20),
            experience_level_category VARCHAR(50),
            company_size_category VARCHAR(20),
            remote_category VARCHAR(20),
            salary_normalized FLOAT
        )
        """)
        
        salary_records = [tuple(x for x in record.values()) for record in salary_df.to_dict('records')]
        salary_columns = salary_df.columns.tolist()
        
        salary_insert = f"""
        INSERT INTO employee_salaries ({','.join(salary_columns)})
        VALUES ({','.join(['%s']*len(salary_columns))})
        """
        
        execute_batch(cur, salary_insert, salary_records)
        
        # 2. Mental health data
        cur.execute("""
        DROP TABLE IF EXISTS mental_health_survey;
        CREATE TABLE mental_health_survey (
            index BIGINT,
            Timestamp VARCHAR(100),
            Age BIGINT,
            Gender VARCHAR(50),
            Country VARCHAR(100),
            state VARCHAR(50),
            self_employed VARCHAR(50),
            family_history VARCHAR(50),
            treatment VARCHAR(50),
            work_interfere VARCHAR(50),
            no_employees VARCHAR(50),
            remote_work VARCHAR(50),
            tech_company VARCHAR(50),
            benefits VARCHAR(50),
            care_options VARCHAR(50),
            wellness_program VARCHAR(50),
            seek_help VARCHAR(50),
            anonymity VARCHAR(50),
            leave VARCHAR(50),
            mental_health_consequence VARCHAR(50),
            phys_health_consequence VARCHAR(50),
            coworkers VARCHAR(50),
            supervisor VARCHAR(50),
            mental_health_interview VARCHAR(50),
            phys_health_interview VARCHAR(50),
            mental_vs_physical VARCHAR(50),
            obs_consequence VARCHAR(50),
            comments TEXT
        )
        """)
        
        mental_health_records = [tuple(x for x in record.values()) for record in mental_health_df.to_dict('records')]
        mental_health_columns = mental_health_df.columns.tolist()
        
        mental_health_insert = f"""
        INSERT INTO mental_health_survey ({','.join(mental_health_columns)})
        VALUES ({','.join(['%s']*len(mental_health_columns))})
        """
        
        execute_batch(cur, mental_health_insert, mental_health_records)
        
        # 3. Employee data
        cur.execute("""
        DROP TABLE IF EXISTS employee_details;
        CREATE TABLE employee_details (
            Employee_ID VARCHAR(50),
            Age BIGINT,
            Gender VARCHAR(50),
            Job_Role VARCHAR(100),
            Industry VARCHAR(100),
            Years_of_Experience BIGINT,
            Work_Location VARCHAR(50),
            Hours_Worked_Per_Week BIGINT,
            Number_of_Virtual_Meetings BIGINT,
            Work_Life_Balance_Rating BIGINT,
            Stress_Level VARCHAR(50),
            Mental_Health_Condition VARCHAR(100),
            Access_to_Mental_Health_Resources VARCHAR(50),
            Productivity_Change VARCHAR(50),
            Social_Isolation_Rating BIGINT,
            Satisfaction_with_Remote_Work VARCHAR(50),
            Company_Support_for_Remote_Work BIGINT,
            Physical_Activity VARCHAR(50),
            Sleep_Quality VARCHAR(50),
            Region VARCHAR(100),
            age_group VARCHAR(10)
        )
        """)
        
        employee_records = [tuple(x for x in record.values()) for record in employee_df.to_dict('records')]
        employee_columns = employee_df.columns.tolist()
        
        employee_insert = f"""
        INSERT INTO employee_details ({','.join(employee_columns)})
        VALUES ({','.join(['%s']*len(employee_columns))})
        """
        
        execute_batch(cur, employee_insert, employee_records)
        
        conn.commit()
        
        context.log.info(f"Loaded {len(salary_records)} salary records, {len(mental_health_records)} mental health records, and {len(employee_records)} employee records to PostgreSQL")
        return {"status": "All data loaded to PostgreSQL"}
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Error loading to PostgreSQL: {str(e)}")
    finally:
        if conn:
            conn.close()

# --- Analysis and Visualization ---

@op
def analyze_data(context, load_result):
    """Perform analysis on structured data in PostgreSQL"""
    conn = None
    try:
        from sqlalchemy import create_engine
        engine = create_engine('postgresql://postgres:123@localhost:5432/postgres')
        
        # 1. Salary analysis
        salary_exp = pd.read_sql("""
            SELECT experience_level_category, 
                   AVG(salary_in_usd) as avg_salary,
                   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary_in_usd) as median_salary
            FROM employee_salaries
            GROUP BY experience_level_category
            ORDER BY avg_salary DESC
        """, engine)
        
        salary_job = pd.read_sql("""
            SELECT job_title, 
                   AVG(salary_in_usd) as avg_salary,
                   COUNT(*) as count
            FROM employee_salaries
            GROUP BY job_title
            HAVING COUNT(*) > 5
            ORDER BY avg_salary DESC
            LIMIT 10
        """, engine)
        
        remote_work = pd.read_sql("""
            SELECT company_size_category, 
                   remote_category,
                   COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY company_size_category), 1) as percentage
            FROM employee_salaries
            GROUP BY company_size_category, remote_category
            ORDER BY company_size_category, percentage DESC
        """, engine)
        
        # 2. Mental health analysis
        mental_health_treatment = pd.read_sql("""
            SELECT treatment, COUNT(*) as count
            FROM mental_health_survey
            GROUP BY treatment
        """, engine)
        
        work_interfere = pd.read_sql("""
            SELECT work_interfere, COUNT(*) as count
            FROM mental_health_survey
            WHERE work_interfere IS NOT NULL
            GROUP BY work_interfere
            ORDER BY count DESC
        """, engine)
        
        # 3. Employee data analysis
        mental_health_by_industry = pd.read_sql("""
            SELECT Industry as industry,
                   COUNT(*) as total_employees,
                   SUM(CASE WHEN Mental_Health_Condition != 'No condition' THEN 1 ELSE 0 END) as with_condition,
                   ROUND(SUM(CASE WHEN Mental_Health_Condition != 'No condition' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as percentage
            FROM employee_details
            GROUP BY Industry
            ORDER BY percentage DESC
        """, engine)
        
        remote_satisfaction = pd.read_sql("""
            SELECT Work_Location AS work_location, 
                   Satisfaction_with_Remote_Work AS satisfaction_with_remote_work,
                   COUNT(*) as count
            FROM employee_details
            WHERE work_location IN ('Remote', 'Hybrid', 'Onsite')
            GROUP BY work_location, satisfaction_with_remote_work
            ORDER BY Work_Location, count DESC
        """, engine)
        
        # Additional analysis for problem statement
        # Stress by job role
        stress_by_role = pd.read_sql("""
            SELECT Job_Role, Stress_Level, COUNT(*) as count
            FROM employee_details
            GROUP BY Job_Role, Stress_Level
            ORDER BY Job_Role, count DESC
        """, engine)
        
        # Productivity vs mental health
        productivity_analysis = pd.read_sql("""
            SELECT Mental_Health_Condition, 
                   Productivity_Change,
                   COUNT(*) as count
            FROM employee_details
            GROUP BY Mental_Health_Condition, Productivity_Change
            ORDER BY Mental_Health_Condition, count DESC
        """, engine)
        
        # Mental health by age group
        mental_health_by_age = pd.read_sql("""
            SELECT age_group,
                   COUNT(*) as total,
                   SUM(CASE WHEN Mental_Health_Condition != 'No condition' THEN 1 ELSE 0 END) as with_condition,
                   ROUND(SUM(CASE WHEN Mental_Health_Condition != 'No condition' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as percentage
            FROM employee_details
            GROUP BY age_group
            ORDER BY age_group
        """, engine)
        
        # Correlation data
        correlation_data = pd.read_sql("""
            SELECT 
                Work_Life_Balance_Rating,
                Company_Support_for_Remote_Work,
                Social_Isolation_Rating,
                (CASE WHEN Mental_Health_Condition != 'No condition' THEN 1 ELSE 0 END) as has_condition,
                (CASE WHEN Stress_Level = 'High' THEN 1 ELSE 0 END) as high_stress,
                Hours_Worked_Per_Week
            FROM employee_details
        """, engine)
        
        # Calculate correlation matrix
        correlation_matrix = correlation_data.corr()
        
        return {
            "salary_by_exp": salary_exp,
            "salary_by_job": salary_job,
            "remote_work_dist": remote_work,
            "treatment_stats": mental_health_treatment,
            "work_interfere": work_interfere,
            "mental_health_by_industry": mental_health_by_industry,
            "remote_satisfaction": remote_satisfaction,
            "stress_by_role": stress_by_role,
            "productivity_analysis": productivity_analysis,
            "mental_health_by_age": mental_health_by_age,
            "correlation_matrix": correlation_matrix,
            "status": "Analysis complete"
        }
    except Exception as e:
        raise Exception(f"Analysis error: {str(e)}")
    finally:
        if conn:
            conn.close()

@op
def create_visualizations(context, analysis_results):
    """Create visualizations with matplotlib fallback"""
    try:
        context.log.info("Starting visualization creation...")
        
        # Create output directories
        output_dirs = ["visualizations", "dashboard", "report_images"]
        for dir_name in output_dirs:
            dir_path = Path(dir_name)
            dir_path.mkdir(exist_ok=True)
            context.log.info(f"Created directory: {dir_path.absolute()}")
        
        results = {}
        
        # 1. Salary Visualizations
        context.log.info("Creating salary visualizations...")
        fig1 = px.bar(analysis_results["salary_by_exp"],
                     x='experience_level_category',
                     y='avg_salary',
                     title='Average Salary by Experience Level')
        
        fig1_path = Path("visualizations/salary_by_exp.html").absolute()
        fig1.write_html(fig1_path)
        img1_path = Path("report_images/salary_by_exp.png").absolute()
        if not save_plotly_figure(fig1, img1_path, context):
            raise RuntimeError("Failed to save salary by experience image")
        results["salary_by_exp"] = str(img1_path)
        analysis_results["salary_by_exp"].to_csv("report_images/salary_by_exp_data.csv", index=False)
        
        fig2 = px.bar(analysis_results["salary_by_job"],
                     x='job_title',
                     y='avg_salary',
                     color='count',
                     title='Top Paying Job Titles')
        
        fig2_path = Path("visualizations/top_jobs.html").absolute()
        fig2.write_html(fig2_path)
        img2_path = Path("report_images/top_jobs.png").absolute()
        if not save_plotly_figure(fig2, img2_path, context):
            raise RuntimeError("Failed to save top jobs image")
        results["top_jobs"] = str(img2_path)
        analysis_results["salary_by_job"].to_csv("report_images/salary_by_job_data.csv", index=False)
        
        fig3 = px.sunburst(analysis_results["remote_work_dist"],
                          path=['company_size_category', 'remote_category'],
                          values='percentage',
                          title='Remote Work Distribution by Company Size')
        
        fig3_path = Path("visualizations/remote_work.html").absolute()
        fig3.write_html(fig3_path)
        img3_path = Path("report_images/remote_work.png").absolute()
        if not save_plotly_figure(fig3, img3_path, context):
            raise RuntimeError("Failed to save remote work image")
        results["remote_work"] = str(img3_path)
        analysis_results["remote_work_dist"].to_csv("report_images/remote_work_dist_data.csv", index=False)
        
        # 2. Mental Health Visualizations
        context.log.info("Creating mental health visualizations...")
        fig4 = px.pie(analysis_results["treatment_stats"],
                     names='treatment',
                     values='count',
                     title='Mental Health Treatment Seeking Behavior')
        
        fig4_path = Path("visualizations/treatment_stats.html").absolute()
        fig4.write_html(fig4_path)
        img4_path = Path("report_images/treatment_stats.png").absolute()
        if not save_plotly_figure(fig4, img4_path, context):
            raise RuntimeError("Failed to save treatment stats image")
        results["treatment_stats"] = str(img4_path)
        analysis_results["treatment_stats"].to_csv("report_images/treatment_stats_data.csv", index=False)
        
        fig5 = px.bar(analysis_results["work_interfere"],
                     x='work_interfere',
                     y='count',
                     title='How Often Mental Health Interferes with Work')
        
        fig5_path = Path("visualizations/work_interfere.html").absolute()
        fig5.write_html(fig5_path)
        img5_path = Path("report_images/work_interfere.png").absolute()
        if not save_plotly_figure(fig5, img5_path, context):
            raise RuntimeError("Failed to save work interfere image")
        results["work_interfere"] = str(img5_path)
        analysis_results["work_interfere"].to_csv("report_images/work_interfere_data.csv", index=False)
        
        # 3. Employee Data Visualizations
        context.log.info("Creating employee data visualizations...")
        fig6 = px.bar(analysis_results["mental_health_by_industry"],
                     x='industry',
                     y='percentage',
                     title='Percentage of Employees with Mental Health Conditions by Industry')
        
        fig6_path = Path("visualizations/mental_health_industry.html").absolute()
        fig6.write_html(fig6_path)
        img6_path = Path("report_images/mental_health_industry.png").absolute()
        if not save_plotly_figure(fig6, img6_path, context):
            raise RuntimeError("Failed to save mental health by industry image")
        results["mental_health_industry"] = str(img6_path)
        analysis_results["mental_health_by_industry"].to_csv("report_images/mental_health_by_industry_data.csv", index=False)
        
        fig7 = px.bar(analysis_results["remote_satisfaction"],
                     x='work_location',
                     y='count',
                     color='satisfaction_with_remote_work',
                     title='Remote Work Satisfaction by Work Location')
        
        fig7_path = Path("visualizations/remote_satisfaction.html").absolute()
        fig7.write_html(fig7_path)
        img7_path = Path("report_images/remote_satisfaction.png").absolute()
        if not save_plotly_figure(fig7, img7_path, context):
            raise RuntimeError("Failed to save remote satisfaction image")
        results["remote_satisfaction"] = str(img7_path)
        analysis_results["remote_satisfaction"].to_csv("report_images/remote_satisfaction_data.csv", index=False)
        
        # Additional visualizations for problem statement
        # Stress by job role
        fig8 = px.bar(analysis_results["stress_by_role"],
                     x='job_role',
                     y='count',
                     color='stress_level',
                     title='Stress Levels by Job Role')
        
        fig8_path = Path("visualizations/stress_by_role.html").absolute()
        fig8.write_html(fig8_path)
        img8_path = Path("report_images/stress_by_role.png").absolute()
        if not save_plotly_figure(fig8, img8_path, context):
            raise RuntimeError("Failed to save stress by role image")
        results["stress_by_role"] = str(img8_path)
        analysis_results["stress_by_role"].to_csv("report_images/stress_by_role_data.csv", index=False)
        
        # Productivity vs mental health
        fig9 = px.bar(analysis_results["productivity_analysis"],
                     x='mental_health_condition',
                     y='count',
                     color='productivity_change',
                     title='Productivity Change by Mental Health Condition')
        
        fig9_path = Path("visualizations/productivity_analysis.html").absolute()
        fig9.write_html(fig9_path)
        img9_path = Path("report_images/productivity_analysis.png").absolute()
        if not save_plotly_figure(fig9, img9_path, context):
            raise RuntimeError("Failed to save productivity analysis image")
        results["productivity_analysis"] = str(img9_path)
        analysis_results["productivity_analysis"].to_csv("report_images/productivity_analysis_data.csv", index=False)
        
        # Mental health by age group
        fig10 = px.bar(analysis_results["mental_health_by_age"],
                      x='age_group',
                      y='percentage',
                      title='Mental Health Conditions by Age Group')
        
        fig10_path = Path("visualizations/mental_health_age.html").absolute()
        fig10.write_html(fig10_path)
        img10_path = Path("report_images/mental_health_age.png").absolute()
        if not save_plotly_figure(fig10, img10_path, context):
            raise RuntimeError("Failed to save mental health by age image")
        results["mental_health_age"] = str(img10_path)
        analysis_results["mental_health_by_age"].to_csv("report_images/mental_health_by_age_data.csv", index=False)
        
        # Correlation matrix heatmap
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            analysis_results["correlation_matrix"], 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt=".2f",  # Format to 2 decimal places
            annot_kws={"size": 10},  # Adjust annotation size
            linewidths=0.5,  # Add lines between cells
            linecolor='white'  # Line color
        )
        plt.title('Correlation Matrix of Workplace Factors', pad=20)
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.yticks(rotation=0)  # Keep y-axis labels horizontal
        plt.tight_layout()

        # Save the figure
        img11_path = Path("report_images/correlation_matrix.png").absolute()
        plt.savefig(img11_path, bbox_inches='tight', dpi=300)
        plt.close()

        # Save the data with proper index naming
        corr_df = analysis_results["correlation_matrix"]
        corr_df.index.name = 'Variables'
        corr_df.to_csv("report_images/correlation_matrix_data.csv", index=True)
        results["correlation_matrix"] = str(img11_path)
        
        # Create dashboard.py with all visualizations
        dashboard_code = """
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

def create_dashboard():
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        report_images_dir = os.path.join(current_dir, '../report_images')
        
        # Load all data files with correct paths
        def load_data(filename):
            return pd.read_csv(os.path.join(report_images_dir, filename))
            
        salary_exp = load_data('salary_by_exp_data.csv')
        salary_job = load_data('salary_by_job_data.csv')
        remote_work = load_data('remote_work_dist_data.csv')
        treatment_stats = load_data('treatment_stats_data.csv')
        work_interfere = load_data('work_interfere_data.csv')
        mental_health_industry = load_data('mental_health_by_industry_data.csv')
        remote_satisfaction = load_data('remote_satisfaction_data.csv')
        stress_by_role = load_data('stress_by_role_data.csv')
        productivity_analysis = load_data('productivity_analysis_data.csv')
        mental_health_age = load_data('mental_health_by_age_data.csv')
        correlation_matrix = load_data('correlation_matrix_data.csv')

        # Create all figures
        fig1 = px.bar(salary_exp,
                     x='experience_level_category',
                     y='avg_salary',
                     title='Average Salary by Experience Level')

        fig2 = px.bar(salary_job,
                     x='job_title',
                     y='avg_salary',
                     color='count',
                     title='Top Paying Job Titles')

        fig3 = px.sunburst(remote_work,
                          path=['company_size_category', 'remote_category'],
                          values='percentage',
                          title='Remote Work Distribution by Company Size')

        fig4 = px.pie(treatment_stats,
                     names='treatment',
                     values='count',
                     title='Mental Health Treatment Seeking Behavior')

        fig5 = px.bar(work_interfere,
                     x='work_interfere',
                     y='count',
                     title='How Often Mental Health Interferes with Work')

        fig6 = px.bar(mental_health_industry,
                     x='industry',
                     y='percentage',
                     title='Mental Health Conditions by Industry')

        fig7 = px.bar(remote_satisfaction,
                     x='work_location',
                     y='count',
                     color='satisfaction_with_remote_work',
                     title='Remote Work Satisfaction by Location')

        fig8 = px.bar(stress_by_role,
                     x='job_role',
                     y='count',
                     color='stress_level',
                     title='Stress Levels by Job Role')

        fig9 = px.bar(productivity_analysis,
                     x='mental_health_condition',
                     y='count',
                     color='productivity_change',
                     title='Productivity Change by Mental Health Condition')

        fig10 = px.bar(mental_health_age,
                      x='age_group',
                      y='percentage',
                      title='Mental Health Conditions by Age Group')

        # Create correlation matrix plot
        try:
            # Handle different possible formats of the correlation matrix
            if 'Unnamed: 0' in correlation_matrix.columns:
                corr_data = correlation_matrix.set_index('Unnamed: 0')
            elif 'Variables' in correlation_matrix.columns:
                corr_data = correlation_matrix.set_index('Variables')
            else:
                corr_data = correlation_matrix
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                corr_data,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt=".2f",
                annot_kws={"size": 10},
                linewidths=0.5,
                linecolor='white'
            )
            plt.title('Correlation Matrix of Workplace Factors', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', dpi=300)
            plt.close()
            encoded_corr = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            print(f"Error creating correlation matrix: {str(e)}")
            encoded_corr = ""

        # Set up layout with tabs
        app.layout = dbc.Container([
            html.H1("Integrated Workforce Analytics Dashboard", className="mb-4 text-center"),
            
            dcc.Tabs([
                # Tab 1: Salary Data
                dcc.Tab(label='Salary Analysis', children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=fig1), width=6),
                        dbc.Col(dcc.Graph(figure=fig2), width=6)
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=fig3), width=12)
                    ])
                ]),
                
                # Tab 2: Mental Health
                dcc.Tab(label='Mental Health', children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=fig4), width=6),
                        dbc.Col(dcc.Graph(figure=fig5), width=6)
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=fig6), width=12)
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=fig8), width=6),
                        dbc.Col(dcc.Graph(figure=fig10), width=6)
                    ])
                ]),
                
                # Tab 3: Employee Analytics
                dcc.Tab(label='Employee Analytics', children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=fig7), width=6),
                        dbc.Col(dcc.Graph(figure=fig9), width=6)
                    ]),
                    dbc.Row([
                        dbc.Col(html.Img(src=f"data:image/png;base64,{encoded_corr}"), width=12)
                    ])
                ])
            ])
        ], fluid=True)
    except Exception as e:
        app.layout = html.Div([
            html.H1("Error Loading Dashboard"),
            html.P(str(e)),
            html.P("Please run the pipeline first to generate required data files.")
        ])
    
    return app

if __name__ == '__main__':
    app = create_dashboard()
    app.run(debug=True)
"""
        dashboard_path = Path("dashboard/dashboard.py").absolute()
        with open(dashboard_path, "w") as f:
            f.write(dashboard_code)
        context.log.info(f"Created dashboard at {dashboard_path}")
        
        return {
            "status": "Visualizations created successfully",
            "image_paths": results
        }
    except Exception as e:
        context.log.error(f"Error in create_visualizations: {str(e)}")
        context.log.error(traceback.format_exc())
        raise

@op
def generate_report(context, visualizations):
    """Generate PDF report with visualizations"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.cell(200, 10, txt="Integrated Workforce Analytics Report", ln=1, align='C')
        pdf.ln(10)
        
        # Section 1: Salary Data
        pdf.set_font("Arial", size=10, style='B')
        pdf.cell(200, 10, txt="1. Salary Analysis", ln=1)
        
        pdf.cell(200, 10, txt="1.1 Salary by Experience Level", ln=1)
        pdf.image(visualizations["image_paths"]["salary_by_exp"], x=10, w=180)
        pdf.ln(5)
        
        pdf.cell(200, 10, txt="1.2 Top Paying Job Titles", ln=1)
        pdf.image(visualizations["image_paths"]["top_jobs"], x=10, w=180)
        pdf.ln(5)
        
        pdf.cell(200, 10, txt="1.3 Remote Work Distribution", ln=1)
        pdf.image(visualizations["image_paths"]["remote_work"], x=10, w=180)
        pdf.ln(10)
        
        # Section 2: Mental Health
        pdf.set_font("Arial", size=10, style='B')
        pdf.cell(200, 10, txt="2. Mental Health Insights", ln=1)
        
        pdf.cell(200, 10, txt="2.1 Treatment Seeking Behavior", ln=1)
        pdf.image(visualizations["image_paths"]["treatment_stats"], x=10, w=180)
        pdf.ln(5)
        
        pdf.cell(200, 10, txt="2.2 Work Interference", ln=1)
        pdf.image(visualizations["image_paths"]["work_interfere"], x=10, w=180)
        pdf.ln(5)
        
        pdf.cell(200, 10, txt="2.3 Mental Health by Industry", ln=1)
        pdf.image(visualizations["image_paths"]["mental_health_industry"], x=10, w=180)
        pdf.ln(5)
        
        pdf.cell(200, 10, txt="2.4 Stress Levels by Job Role", ln=1)
        pdf.image(visualizations["image_paths"]["stress_by_role"], x=10, w=180)
        pdf.ln(5)
        
        pdf.cell(200, 10, txt="2.5 Mental Health by Age Group", ln=1)
        pdf.image(visualizations["image_paths"]["mental_health_age"], x=10, w=180)
        pdf.ln(10)
        
        # Section 3: Employee Data
        pdf.set_font("Arial", size=10, style='B')
        pdf.cell(200, 10, txt="3. Employee Analytics", ln=1)
        
        pdf.cell(200, 10, txt="3.1 Remote Work Satisfaction", ln=1)
        pdf.image(visualizations["image_paths"]["remote_satisfaction"], x=10, w=180)
        pdf.ln(5)
        
        pdf.cell(200, 10, txt="3.2 Productivity Change by Mental Health", ln=1)
        pdf.image(visualizations["image_paths"]["productivity_analysis"], x=10, w=180)
        pdf.ln(5)
        
        pdf.cell(200, 10, txt="3.3 Correlation Matrix of Workplace Factors", ln=1)
        pdf.image(visualizations["image_paths"]["correlation_matrix"], x=10, w=180)
        
        report_path = "integrated_workforce_report.pdf"
        pdf.output(report_path)
        
        return {"status": "Report generated", "report_path": report_path}
    except Exception as e:
        context.log.error(f"Report generation error: {str(e)}")
        raise

@job
def workforce_analytics_pipeline():
    """End-to-end workforce analytics pipeline"""
    # Extract and store all data sources with multiple outputs
    csv_status, d1_count, d2_count = extract_and_store_csv_data()
    json_status, json_count = extract_and_store_json()
    
    # Process all data
    raw_data = extract_from_mongodb(csv_status, d1_count, d2_count, json_status, json_count)
    transformed_data = transform_data(raw_data)
    load_result = load_to_postgres(transformed_data)
    analysis_results = analyze_data(load_result)
    visualizations = create_visualizations(analysis_results)
    generate_report(visualizations)

if __name__ == "__main__":
    # Create required directories
    Path("visualizations").mkdir(exist_ok=True)
    Path("dashboard").mkdir(exist_ok=True)
    Path("report_images").mkdir(exist_ok=True)
    
    # Execute the pipeline
    instance = DagsterInstance.get()
    recon_job = reconstructable(workforce_analytics_pipeline)
    result = execute_job(recon_job, instance=instance)
    
    if result.success:
        print("\nPipeline executed successfully!")
        print("Outputs created:")
        print("- Visualizations: visualizations/")
        print("- Dashboard components: dashboard/")
        print("- Report images: report_images/")
        print("- PDF report: integrated_workforce_report.pdf")
        
        print("\nTo view the dashboard:")
        print("1. cd dashboard")
        print("2. python dashboard.py")
        print("3. Open http://localhost:8050 in your browser\n")
    else:
        print("Pipeline execution failed")
        print("Check the Dagster logs for error details")