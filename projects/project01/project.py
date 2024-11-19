# project.py


import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta


import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades_df):
    """
    Extract assignment names from the grades DataFrame and categorize them.

    Args:
        grades_df (pd.DataFrame): DataFrame containing grade information.

    Returns:
        dict: A dictionary where keys are syllabus assignment categories \
        and values are lists of FULL (?) assignment names.
    """
    categories = {
        'lab': [],
        'project': [],
        'midterm': [],
        'final': [],
        'disc': [],
        'checkpoint': []
    }

    for column in grades_df.columns:
        parts = column.split()
        if len(parts) > 1:
            continue  # Skip columns with additional information (e.g., "Max Points", "Lateness")
        
        assignment = parts[0]
        if assignment.startswith('lab'):
            categories['lab'].append(assignment)
        elif assignment.startswith('project'):
            categories['project'].append(assignment)
        elif assignment.startswith('Midterm'):
            categories['midterm'].append(assignment)
        elif assignment.startswith('Final'):
            categories['final'].append(assignment)
        elif assignment.startswith('discussion'):
            categories['disc'].append(assignment)
        elif assignment.startswith('checkpoint'):
            categories['checkpoint'].append(assignment)

    return categories
    
    
    
    
# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def projects_total(grades):
    """
    Calculate the total project score for each student, handling both autograded and 
    free response portions correctly.
    
    Args:
        grades (pd.DataFrame): DataFrame containing all grade information
        
    Returns:
        pd.Series: Series containing the average project score (0-1) for each student
    """
    # Step 1: Identify project columns (excluding checkpoints and lateness)
    project_cols = [
        col for col in grades.columns 
        if col.startswith('project') 
        and 'checkpoint' not in col.lower() 
        and 'lateness' not in col.lower()
    ]
    
    # Step 2: Group columns by project number
    project_groups = {}
    for col in project_cols:
        # Extract project number (e.g., "project1" or "project01")
        if '_' in col:  # Handle free response columns
            project_num = col.split('project')[1].split('_')[0]
        else:
            project_num = col.split('project')[1].split()[0]
        
        if project_num not in project_groups:
            project_groups[project_num] = []
        project_groups[project_num].append(col)
    
    # Step 3: Calculate individual project scores
    project_scores = []
    
    for project_num, cols in project_groups.items():
        # Separate columns by type
        auto_cols = [col for col in cols if 'free_response' not in col and 'Max Points' not in col]
        frq_cols = [col for col in cols if 'free_response' in col and 'Max Points' not in col]
        
        # Get corresponding max points columns
        auto_max_cols = [f"{col} - Max Points" for col in auto_cols if f"{col} - Max Points" in grades.columns]
        frq_max_cols = [f"{col} - Max Points" for col in frq_cols if f"{col} - Max Points" in grades.columns]
        
        # Calculate autograded portion
        auto_score = grades[auto_cols].fillna(0).sum(axis=1)
        auto_max = grades[auto_max_cols].fillna(0).sum(axis=1)
        
        # Calculate free response portion if it exists
        frq_score = grades[frq_cols].fillna(0).sum(axis=1)
        frq_max = grades[frq_max_cols].fillna(0).sum(axis=1)
        
        # Combine scores
        total_score = auto_score + frq_score
        total_max = auto_max + frq_max
        
        # Calculate normalized score (0-1)
        project_grade = np.where(total_max > 0, total_score / total_max, 0)
        project_scores.append(pd.Series(project_grade, name=f'Project {project_num}'))
    
    # Step 4: Calculate average score across all projects
    all_projects = pd.concat(project_scores, axis=1)
    total_project_score = all_projects.mean(axis=1)
    
    return total_project_score.clip(0, 1)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col):
    def parse_time(time_str):
        # Handle NaN or non-string values gracefully
        if isinstance(time_str, str):
            parts = time_str.split(':')
            return timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2]))
        return timedelta()  # Default to zero lateness if the input is not valid

    grace_period = timedelta(hours=2)
    one_week = timedelta(days=7)
    two_weeks = timedelta(days=14)

    def calculate_multiplier(lateness):
        if lateness <= grace_period:
            return 1.0
        elif lateness <= one_week:
            return 0.9
        elif lateness <= two_weeks:
            return 0.7
        else:
            return 0.4

    # Apply the logic to each element in the column
    return col.apply(lambda x: calculate_multiplier(parse_time(x)))


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def process_labs(grades):
    """
    Process lab scores by adjusting for lateness and normalizing to a score between 0 and 1.

    Args:
        grades (pd.DataFrame): DataFrame containing grade information, including lab scores and lateness.

    Returns:
        pd.DataFrame: DataFrame of processed lab scores, adjusted for lateness, with scores normalized between 0 and 1.
    """
    lab_cols = [col for col in grades.columns if col.startswith('lab') and 'Max Points' not in col and 'Lateness' not in col]
    lateness_cols = [col for col in grades.columns if col.startswith('lab') and 'Lateness' in col]
    max_points_cols = [col for col in grades.columns if col.startswith('lab') and 'Max Points' in col]

    processed_labs = pd.DataFrame(index=grades.index)

    for lab_col, lateness_col, max_points_col in zip(lab_cols, lateness_cols, max_points_cols):
        # Apply lateness penalty
        lateness_multiplier = lateness_penalty(grades[lateness_col])

        # Normalize lab score between 0 and 1 based on max points
        normalized_lab_score = grades[lab_col] / grades[max_points_col]

        # Apply lateness multiplier to the normalized score
        adjusted_lab_score = normalized_lab_score * lateness_multiplier

        # Ensure all scores are between 0 and 1, and handle missing values (NaNs)
        adjusted_lab_score = adjusted_lab_score.clip(0, 1).fillna(0)

        # Add the processed lab score to the new DataFrame
        processed_labs[lab_col] = adjusted_lab_score

    return processed_labs




# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def lab_total(processed):
    """
    Calculate the total lab grade after processing lateness penalties and dropping the lowest score.

    Args:
        processed (pd.DataFrame): DataFrame containing processed lab grades.

    Returns:
        pd.Series: Series with the total lab grade for each student as a proportion between 0 and 1.
    """
    lab_scores = [col for col in processed.columns if col.startswith('lab')]
    
    # Drop the lowest lab score and compute the average of the remaining scores
    lab_totals = processed[lab_scores].apply(lambda row: (row.dropna().sum() - row.min()) / (len(row) - 1), axis=1)
    
    return lab_totals




# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------




def total_points(grades):
    """
    Calculate total course grade as a proportion between 0 and 1 using Professor Yutian's syllabus weights:
    - Labs: 20%
    - Projects: 30%
    - Checkpoints: 2.5%
    - Discussions: 2.5%
    - Midterm: 15%
    - Final: 30%
    
    Args:
        grades (pd.DataFrame): DataFrame containing all grade information
        
    Returns:
        pd.Series: Series containing each student's course grade as a proportion between 0 and 1
    """
    # Convert columns to numeric and fill NaN with 0
    grades = grades.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    def get_component_score(prefix):
        """Helper function to calculate normalized scores for any assignment type"""
        cols = [col for col in grades.columns if col.startswith(prefix) and 'Max Points' not in col]
        max_cols = [col for col in grades.columns if col.startswith(prefix) and 'Max Points' in col]
        
        if not cols or not max_cols:
            return 0
        
        scores = grades[cols].sum(axis=1)
        max_points = grades[max_cols].sum(axis=1)
        return np.where(max_points > 0, scores / max_points, 0).clip(0, 1)
    
    def get_exam_score(exam_name):
        """Helper function to calculate normalized scores for exams"""
        col = next((col for col in grades.columns if exam_name in col and 'Max Points' not in col), None)
        max_col = f"{col} - Max Points" if col else None
        
        if not (col and max_col in grades.columns):
            return 0
            
        return np.where(grades[max_col] > 0, 
                       grades[col] / grades[max_col], 
                       0).clip(0, 1)
    
    # Calculate component scores with corrected weights from Professor Yutian's syllabus
    component_weights = {
        'lab': (lab_total(process_labs(grades)), 0.20),      # 20% for labs
        'project': (projects_total(grades), 0.30),           # 30% for projects
        'discussion': (get_component_score('discussion'), 0.025),  # 2.5% for discussions
        'checkpoint': (get_component_score('checkpoint'), 0.025),  # 2.5% for checkpoints
        'midterm': (get_exam_score('Midterm'), 0.15),       # 15% for midterm
        'final': (get_exam_score('Final'), 0.30)            # 30% for final
    }
    
    # Calculate weighted sum
    course_grade = sum(score * weight for score, weight in component_weights.values())
    
    return course_grade.clip(0, 1)
# Example usage
# out = total_points(grades)
# print(out.mean())  # Should be between 0.7 and 0.9
# print(bool(0.7 < out.mean() < 0.9))


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------




def final_grades(grades):
    """
    Takes in a Series of final course grades and returns a Series of letter grades
    based on the specified cutoffs.
    """
    def grade_to_letter(grade):
        if grade >= 0.9:
            return 'A'
        elif grade >= 0.8:
            return 'B'
        elif grade >= 0.7:
            return 'C'
        elif grade >= 0.6:
            return 'D'
        else:
            return 'F'
    
    return grades.apply(grade_to_letter)

def letter_proportions(grades):
    """
    Takes in a Series of final course grades and returns a Series with the proportion
    of the class that received each letter grade, sorted by grade value (B, C, A, D, F).
    """
    # Convert numerical grades to letter grades
    letter_grades = final_grades(grades)
    
    # Calculate the proportions
    proportions = letter_grades.value_counts(normalize=True)
    
    # Create a Series with the expected letter grades in the specific order
    ordered_grades = pd.Series(0.0, index=['B', 'C', 'A', 'D', 'F'])
    
    # Update the ordered grades with actual proportions
    for grade in proportions.index:
        if grade in ordered_grades.index:
            ordered_grades[grade] = proportions[grade]
    
    return ordered_grades
# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------



def raw_redemption(final_breakdown, redemption_questions):
    # Extract PID and redemption question columns
    redemption_cols = ['PID'] + [col for col in final_breakdown.columns if any(f'Question {q}' in col for q in redemption_questions)]
    redemption_df = final_breakdown[redemption_cols].copy()
    
    # Extract max points for each question
    max_points = {}
    for col in redemption_df.columns[1:]:
        max_points[col] = float(col.split('(')[-1].split()[0])
    
    # Calculate total points earned and total possible points
    total_earned = redemption_df.iloc[:, 1:].sum(axis=1)
    total_possible = sum(max_points.values())
    
    # Calculate raw redemption score
    raw_score = total_earned / total_possible
    
    # Create result DataFrame
    result = pd.DataFrame({
        'PID': redemption_df['PID'],
        'Raw Redemption Score': raw_score
    })
    
    # Handle students who didn't take the final
    result.loc[final_breakdown.iloc[:, 1:].isnull().all(axis=1), 'Raw Redemption Score'] = 0
    
    return result

def combine_grades(grades, redemption_scores):
    # Merge grades with redemption scores based on PID
    combined = grades.merge(redemption_scores, on='PID', how='left')
    
    # Fill NaN values in 'Raw Redemption Score' with 0
    # This handles cases where a student in grades doesn't have a redemption score
    combined['Raw Redemption Score'] = combined['Raw Redemption Score'].fillna(0)
    
    return combined


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------



def z_score(series):
    """
    Convert a series of numbers to z-scores.
    """
    return (series - series.mean()) / series.std(ddof=0)

def add_post_redemption(grades_combined):
    # Extract midterm and raw redemption scores
    midterm_scores = grades_combined['Midterm']
    redemption_scores = grades_combined['Raw Redemption Score']

    # Calculate pre-redemption midterm scores as proportions
    max_midterm_score = midterm_scores.max()
    pre_redemption = midterm_scores / max_midterm_score

    # Handle students who didn't take the midterm
    midterm_taken = ~midterm_scores.isna()
    
    # Calculate z-scores for midterm and redemption
    midterm_z = z_score(midterm_scores[midterm_taken])
    redemption_z = z_score(redemption_scores)

    # Initialize post-redemption scores with pre-redemption scores
    post_redemption = pre_redemption.copy()

    # Apply redemption policy
    mask = (redemption_z > midterm_z) & midterm_taken
    post_redemption[mask] = (pre_redemption[mask] + redemption_scores[mask]) / 2

    # Create new DataFrame with additional columns
    result = grades_combined.copy()
    result['Midterm Score Pre-Redemption'] = pre_redemption
    result['Midterm Score Post-Redemption'] = post_redemption

    return result


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------



def total_points_post_redemption(grades_combined):
    # Get the total points before redemption
    pre_redemption_total = total_points(grades_combined)
    
    # Calculate pre-redemption midterm score
    max_midterm_score = grades_combined['Midterm'].max()
    pre_redemption_midterm = grades_combined['Midterm'] / max_midterm_score
    
    # Calculate post-redemption midterm score
    redemption_score = grades_combined['Raw Redemption Score']
    post_redemption_midterm = np.maximum(pre_redemption_midterm, 
                                         (pre_redemption_midterm + redemption_score) / 2)
    
    # Calculate the difference in midterm scores
    midterm_difference = post_redemption_midterm - pre_redemption_midterm
    
    # Adjust the total points
    # The midterm is worth 15% of the total grade
    post_redemption_total = pre_redemption_total + (midterm_difference * 0.15)
    
    # Ensure the scores are between 0 and 1
    post_redemption_total = post_redemption_total.clip(0, 1)
    
    return post_redemption_total

def proportion_improved(grades_combined):
    # Calculate grades before and after redemption
    pre_redemption_grades = total_points(grades_combined)
    post_redemption_grades = total_points_post_redemption(grades_combined)
    
    # Convert to letter grades
    pre_letters = grade_to_letter(pre_redemption_grades)
    post_letters = grade_to_letter(post_redemption_grades)
    
    # Count how many students improved
    improved = (post_letters > pre_letters).sum()
    
    # Calculate the proportion
    proportion = improved / len(grades_combined)
    
    return proportion

def grade_to_letter(scores):
    def assign_letter(score):
        if score >= 0.90: return 'A'
        elif score >= 0.80: return 'B'
        elif score >= 0.70: return 'C'
        elif score >= 0.60: return 'D'
        else: return 'F'
    
    return scores.apply(assign_letter)


# # ---------------------------------------------------------------------
# # QUESTION 11
# # ---------------------------------------------------------------------


# def section_most_improved(grades_analysis):
#     ...
    
# def top_sections(grades_analysis, t, n):
#     ...


# # ---------------------------------------------------------------------
# # QUESTION 12
# # ---------------------------------------------------------------------


# def rank_by_section(grades_analysis):
#     ...







# # ---------------------------------------------------------------------
# # QUESTION 13
# # ---------------------------------------------------------------------


# def letter_grade_heat_map(grades_analysis):
#     ...
