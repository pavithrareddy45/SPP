"""
=============================================================================
STAGE 4: Personalized Improvement Recommendation Engine
=============================================================================
Module: src/recommendation_engine.py
Project: Student Performance Predictor

This module provides rule-based pedagogical advisory logic completely decoupled
from the Machine Learning model. It evaluates four key academic dimensions:
  1. Performance Tier (Distinction, Merit, Pass, At-Risk)
  2. Grade Momentum (Improving, Plateaued, Declining based on G1 -> G2)
  3. Study Time Health (Realistic, gradual weekly increases; no burnout)
  4. Attendance Health (Educational intervention for high absences)

Pedagogical Design Principle:
While statistical regression models may find idiosyncratic correlations in historical
data, educational recommendations must adhere to sound teaching practices:
  - High absences cause conceptual gaps and require supportive intervention.
  - Study time must increase gradually and realistically (avoid 20-hour cram schedules).
  - Score drops (declining momentum) require targeted diagnostic remediation.
=============================================================================
"""

from typing import Dict, Any, List, Optional


def validate_student_inputs(student_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and normalizes student input features.

    Why this function is important (Beginner Explanation):
    Web users and external forms can submit invalid, missing, or out-of-range
    values (e.g., negative grades or letters instead of numbers). This function
    verifies that every input is within realistic educational bounds before any
    calculations are performed.

    Parameters:
        student_data (dict): Dictionary with keys 'studytime', 'failures',
                             'absences', 'G1', and 'G2'.

    Returns:
        dict: Cleaned and type-converted dictionary of student features.

    Raises:
        ValueError: If any required field is missing, non-numeric, or outside valid bounds.
    """
    required_fields = ["studytime", "failures", "absences", "G1", "G2"]
    
    # 1. Check for missing required fields
    for field in required_fields:
        if field not in student_data or student_data[field] is None:
            raise ValueError(f"Missing required input field: '{field}'")

    # 2. Convert and validate numeric types
    try:
        studytime = int(student_data["studytime"])
        failures = int(student_data["failures"])
        absences = int(student_data["absences"])
        g1 = float(student_data["G1"])
        g2 = float(student_data["G2"])
    except (TypeError, ValueError) as err:
        raise ValueError(f"All student inputs must be valid numbers: {err}")

    # 3. Validate logical educational ranges
    # studytime: 1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h)
    if studytime not in [1, 2, 3, 4]:
        raise ValueError(f"Invalid studytime: {studytime}. Must be 1, 2, 3, or 4.")

    if failures < 0 or failures > 10:
        raise ValueError(f"Invalid failures: {failures}. Must be an integer between 0 and 10.")

    if absences < 0 or absences > 100:
        raise ValueError(f"Invalid absences: {absences}. Must be between 0 and 100 days.")

    if not (0.0 <= g1 <= 20.0):
        raise ValueError(f"Invalid G1 score: {g1}. Must be between 0.0 and 20.0 marks.")

    if not (0.0 <= g2 <= 20.0):
        raise ValueError(f"Invalid G2 score: {g2}. Must be between 0.0 and 20.0 marks.")

    return {
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "G1": g1,
        "G2": g2
    }


def determine_performance_tier(predicted_score: float) -> Dict[str, str]:
    """
    Evaluates the predicted final exam score against standard educational benchmarks.

    Tiers defined:
      - Distinction : predicted_score >= 16.0 (Top honors, exceptional mastery)
      - Merit       : 14.0 <= predicted_score < 16.0 (Strong, well-above average)
      - Pass        : 10.0 <= predicted_score < 14.0 (Satisfactory, meets course requirements)
      - At-Risk     : predicted_score < 10.0 (Below passing threshold, needs urgent support)

    Parameters:
        predicted_score (float): The model's bounded prediction (0 to 20).

    Returns:
        dict: Tier classification containing tier name, visual badge style, and description.
    """
    score = round(float(predicted_score), 2)
    
    if score >= 16.0:
        return {
            "tier": "Distinction",
            "badge_class": "badge-distinction",
            "badge_color": "#16a34a",  # Emerald Green
            "summary": "Outstanding academic standing. On track for top honors."
        }
    elif score >= 14.0:
        return {
            "tier": "Merit",
            "badge_class": "badge-merit",
            "badge_color": "#2563eb",  # Royal Blue
            "summary": "Solid, above-average performance with strong potential."
        }
    elif score >= 10.0:
        return {
            "tier": "Pass",
            "badge_class": "badge-pass",
            "badge_color": "#d97706",  # Amber / Warning
            "summary": "Meeting basic passing standards, but has vulnerable areas."
        }
    else:
        return {
            "tier": "At-Risk",
            "badge_class": "badge-at-risk",
            "badge_color": "#dc2626",  # Crimson Red
            "summary": "Currently below passing standard (10/20). Needs immediate intervention."
        }


def analyze_grade_momentum(g1: float, g2: float) -> Dict[str, Any]:
    """
    Compares Term 1 (G1) and Term 2 (G2) scores to detect learning trajectory.

    Momentum classifications:
      - Improving : G2 - G1 > 1.0 (Student is accelerating academic mastery)
      - Plateaued : -1.0 <= G2 - G1 <= 1.0 (Stable, consistent performance)
      - Declining : G2 - G1 < -1.0 (Performance dropped, indicating learning friction)

    Parameters:
        g1 (float): Term 1 / assignment grade (0 to 20).
        g2 (float): Term 2 midterm exam grade (0 to 20).

    Returns:
        dict: Momentum status, delta value, and pedagogical interpretation.
    """
    delta = round(g2 - g1, 2)

    if delta > 1.0:
        return {
            "status": "Improving",
            "delta": delta,
            "badge_color": "#16a34a",
            "description": f"Positive trajectory: Improved by +{delta:.1f} marks from Term 1 to Term 2.",
            "guidance": "Your active efforts are yielding results. Maintain current study routines."
        }
    elif delta < -1.0:
        return {
            "status": "Declining",
            "delta": delta,
            "badge_color": "#dc2626",
            "description": f"Negative trajectory: Dropped by {abs(delta):.1f} marks between Term 1 and Term 2.",
            "guidance": "Midterm score dropped compared to Term 1. Immediate concept review of Term 2 topics is advised."
        }
    else:
        return {
            "status": "Plateaued",
            "delta": delta,
            "badge_color": "#475569",
            "description": f"Stable performance: Grade remained consistent (change of {delta:+.1f} marks).",
            "guidance": "Consistent results, but you have plateaued. Adopting active testing can unlock higher marks."
        }


def generate_study_recommendations(studytime_level: int, performance_tier: str) -> List[str]:
    """
    Generates realistic, gradual study time advice tailored to current habits.

    Why gradual advice is essential (Pedagogical Rule):
    Telling a student who studies <2 hours/week to suddenly study 20 hours/week
    leads to quick burnout and frustration. We recommend incremental, realistic
    increases of 2-3 hours per week.

    Parameters:
        studytime_level (int): 1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h).
        performance_tier (str): Distinction, Merit, Pass, or At-Risk.

    Returns:
        list of str: Concrete, progressive study habit recommendations.
    """
    recommendations = []

    if studytime_level == 1:
        # Currently studying < 2 hours/week
        recommendations.append(
            "Current study time (< 2 hours/week) is insufficient for exam mastery."
        )
        recommendations.append(
            "Gradual Target: Add 30 to 45 minutes of dedicated study on 3 weekdays (aiming for 3 to 4 hours/week)."
        )
        recommendations.append(
            "Technique: Use the Pomodoro Technique (25 minutes focused study, 5 minutes break) to build stamina."
        )

    elif studytime_level == 2:
        # Currently studying 2 to 5 hours/week
        recommendations.append(
            "Current study time (2-5 hours/week) provides a basic routine, but has room for growth."
        )
        recommendations.append(
            "Gradual Target: Increase study schedule to 1 hour each school day (aiming for 5 to 6 hours/week)."
        )
        recommendations.append(
            "Technique: Shift 50% of study time from passive reading to solving past exam questions."
        )

    elif studytime_level == 3:
        # Currently studying 5 to 10 hours/week
        recommendations.append(
            "Healthy study routine (5-10 hours/week) demonstrated."
        )
        recommendations.append(
            "Optimization: Incorporate spaced repetition and weekly self-quizzing on challenging topics."
        )

    else:
        # Level 4 (> 10 hours/week)
        recommendations.append(
            "High study commitment (> 10 hours/week). Excellent dedication!"
        )
        recommendations.append(
            "Burnout Prevention: Ensure you balance intense study with adequate rest, sleep, and physical activity."
        )

    # Add tier-specific study tip
    if performance_tier == "At-Risk":
        recommendations.append(
            "Priority: Focus first on high-weight core syllabus topics rather than trying to revise everything at once."
        )
    elif performance_tier == "Distinction":
        recommendations.append(
            "Mastery: Attempt advanced competition-level exercises and peer-tutor classmates to solidify knowledge."
        )

    return recommendations


def generate_attendance_recommendations(absences: int) -> List[str]:
    """
    Generates supportive attendance advice.

    Educational Rule Enforced:
    Even if statistical quirks exist in raw data, classroom attendance is
    foundational for student comprehension. This function warns of high absences
    and offers supportive, constructive catch-up steps.

    Parameters:
        absences (int): Number of days/periods absent.

    Returns:
        list of str: Attendance recommendations and mitigation steps.
    """
    recommendations = []

    if absences >= 15:
        recommendations.append(
            f"High Absence Alert: {absences} missed classes significantly threatens syllabus continuity."
        )
        recommendations.append(
            "Immediate Action: Meet with your teacher/academic counselor to map missed lessons."
        )
        recommendations.append(
            "Peer Catch-Up: Pair with a reliable classmate to copy missed lecture notes and homework solutions."
        )
        recommendations.append(
            "Commitment: Aim for zero unexcused absences for the remainder of the term."
        )

    elif absences >= 6:
        recommendations.append(
            f"Moderate Absence Notice: {absences} absences detected."
        )
        recommendations.append(
            "Catch-Up Target: Review class notes for any missed topics before the next examination cycle."
        )
        recommendations.append(
            "Attendance Goal: Prioritize attending all review sessions and lab periods."
        )

    else:
        recommendations.append(
            f"Excellent Attendance: Only {absences} absences recorded."
        )
        recommendations.append(
            "Keep it up! Consistent presence in class ensures you never miss exam tips and discussions."
        )

    return recommendations


def build_overall_improvement_plan(
    tier_info: Dict[str, str],
    momentum_info: Dict[str, Any],
    study_recs: List[str],
    attendance_recs: List[str],
    student_inputs: Dict[str, Any],
    predicted_score: float
) -> List[str]:
    """
    Synthesizes the four dimensions into a prioritized, actionable 3 to 4 step roadmap.

    Parameters:
        tier_info (dict): Output from determine_performance_tier.
        momentum_info (dict): Output from analyze_grade_momentum.
        study_recs (list): Study recommendations.
        attendance_recs (list): Attendance recommendations.
        student_inputs (dict): Validated input parameters.
        predicted_score (float): Model predicted score.

    Returns:
        list of str: 3 to 4 concrete, sequential action items.
    """
    tier = tier_info["tier"]
    momentum = momentum_info["status"]
    absences = student_inputs["absences"]
    failures = student_inputs["failures"]

    plan = []

    # Step 1: Immediate Critical Priority
    if tier == "At-Risk":
        plan.append(
            "Phase 1 (Immediate - Week 1): Schedule an academic intervention meeting with your instructor. "
            "Identify the 3 weakest subject modules responsible for low test scores."
        )
    elif momentum == "Declining":
        plan.append(
            "Phase 1 (Immediate - Week 1): Conduct a post-mortem review of your Term 2 exam. "
            "Identify why performance dropped from Term 1 and correct recurring misconceptions."
        )
    elif tier == "Distinction":
        plan.append(
            "Phase 1 (Week 1): Maintain momentum. Review previous exam papers to identify subtle edge-case "
            "questions and ensure flawless execution on high-difficulty questions."
        )
    else:
        plan.append(
            "Phase 1 (Week 1): Consolidate core notes. Convert class summaries into quick-reference formula "
            "and concept sheets."
        )

    # Step 2: Study Habit Action
    if student_inputs["studytime"] == 1:
        plan.append(
            "Phase 2 (Weeks 2-3): Establish a non-negotiable study block of 45 minutes on Mondays, Wednesdays, "
            "and Fridays. Focus exclusively on active practice problems."
        )
    elif student_inputs["studytime"] == 2:
        plan.append(
            "Phase 2 (Weeks 2-3): Upgrade study sessions from 3 to 5 hours/week. Dedicate weekend slots "
            "specifically to timed mock testing."
        )
    else:
        plan.append(
            "Phase 2 (Weeks 2-3): Engage in spaced repetition. Test yourself on concepts learned 2 weeks ago "
            "to prevent memory decay before the final exam."
        )

    # Step 3: Attendance or Remediation Action
    if absences >= 10:
        plan.append(
            "Phase 3 (Ongoing): Secure perfect attendance for the next 4 weeks. Request supplementary worksheets "
            "for the classes missed earlier in the term."
        )
    elif failures > 0:
        plan.append(
            "Phase 3 (Ongoing): Form a study group or attend weekly school tutoring to address conceptual gaps "
            "carried over from prior course attempts."
        )
    else:
        plan.append(
            "Phase 3 (Ongoing): Complete at least two full-length timed mock exams under exam conditions "
            "to build speed and test endurance."
        )

    # Step 4: Final Target Goal
    target_score = min(20.0, round(predicted_score + 2.5, 1))
    plan.append(
        f"Goal: By following these steps, target closing the gap to reach an exam score of {target_score}/20."
    )

    return plan


def generate_recommendations(
    student_data: Dict[str, Any],
    predicted_score: float
) -> Dict[str, Any]:
    """
    Main entry point for the Personalized Improvement Recommendation Engine.

    Why this function is structured this way (Beginner Explanation):
    It takes raw student inputs and the ML model's predicted final exam score,
    validates everything cleanly, computes the 4 diagnostic dimensions, and returns
    a clean, structured JSON-compatible dictionary ready to render in the web UI.

    Parameters:
        student_data (dict): Student parameters (studytime, failures, absences, G1, G2).
        predicted_score (float): Predicted exam mark (bounded between 0.0 and 20.0).

    Returns:
        dict: Complete structured recommendations package.
    """
    # 1. Validate inputs
    validated_inputs = validate_student_inputs(student_data)

    # 2. Bound predicted score safely within 0 to 20
    bounded_pred = max(0.0, min(20.0, round(float(predicted_score), 2)))

    # 3. Dimension A: Performance Tier
    tier_info = determine_performance_tier(bounded_pred)

    # 4. Dimension B: Grade Momentum
    momentum_info = analyze_grade_momentum(validated_inputs["G1"], validated_inputs["G2"])

    # 5. Dimension C: Study Time Guidance
    study_recs = generate_study_recommendations(validated_inputs["studytime"], tier_info["tier"])

    # 6. Dimension D: Attendance Guidance
    attendance_recs = generate_attendance_recommendations(validated_inputs["absences"])

    # 7. Action Plan Synthesis
    improvement_plan = build_overall_improvement_plan(
        tier_info=tier_info,
        momentum_info=momentum_info,
        study_recs=study_recs,
        attendance_recs=attendance_recs,
        student_inputs=validated_inputs,
        predicted_score=bounded_pred
    )

    # 8. Return structured payload
    return {
        "status": "success",
        "inputs": validated_inputs,
        "predicted_score": bounded_pred,
        "performance_tier": tier_info["tier"],
        "tier_details": tier_info,
        "grade_momentum": momentum_info["status"],
        "momentum_details": momentum_info,
        "study_recommendations": study_recs,
        "attendance_recommendations": attendance_recs,
        "overall_improvement_plan": improvement_plan
    }
