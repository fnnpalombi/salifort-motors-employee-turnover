# Predicting Employee Turnover at Salifort Motors

Salifort Motors was losing employees faster than it wanted to, and every departure is expensive: recruiting, hiring, and training a replacement costs far more than keeping the person who left. This project builds a model that flags employees at risk of leaving, and, more useful for HR day to day, it pins down which factors actually drive people out so the company can act before the resignation letter arrives.

*Salifort Motors is a fictional company and the dataset is synthetic, both provided as the capstone for the Google Advanced Data Analytics certificate. The scenario is a teaching case; the analysis, feature engineering, and modeling choices here are my own.*

## The data

The dataset holds 14,999 employee records from an HR survey, each labeled with whether that person left the company. Dropping duplicate rows removed 3,008 records and left 11,991 unique employees to work with. The fields:

| Field | Description |
|---|---|
| `satisfaction_level` | Self-reported satisfaction (0–1) |
| `last_evaluation` | Score from the most recent performance review (0–1) |
| `number_project` | Number of projects the employee is assigned |
| `average_monthly_hours` | Average hours worked per month |
| `time_with_company` | Years at the company |
| `work_accident` | Whether they had a workplace accident |
| `left` | Whether they left the company (the target) |
| `promotion_last_5years` | Whether they were promoted in the last five years |
| `dept` | Department |
| `salary` | Salary band (low / medium / high) |

Roughly 16% of employees in the data left, so the two classes are imbalanced. That ratio shaped how I trained the model and, just as much, how I scored it: accuracy alone would look great for a model that simply predicted "stayed" every time, so it isn't the metric to trust here.

## Approach

I worked through the project with the PACE framework (Plan, Analyze, Construct, Execute).

**Plan and Analyze.** I started with exploratory analysis to understand who leaves and why. Satisfaction level had the clearest relationship with leaving. A few things stood out that I didn't expect: satisfaction barely moved across salary bands or across departments, so pay level alone wasn't explaining dissatisfaction, and employees who had been promoted in the last five years left at a much lower rate than those who hadn't. I flagged that last one as suggestive rather than proven, since it would need a significance test before anyone leaned on it.

**Construct.** I encoded the categorical fields before modeling: salary as an ordinal rank (low/medium/high), and department with target encoding fit on the training data only, so no information about the test set leaked in. I also engineered a `relative_salary_dept` feature, each employee's salary rank against their department's average, to capture whether someone is paid low *for their team* rather than low overall.

I chose a random forest, which handles the mix of numeric and encoded categorical features well and doesn't need the data scaled. I split the data 70/15/15 into training, validation, and test sets, stratified so the 16:84 class ratio held in each. I tuned the model with a five-fold grid search optimizing F1 (the right target under class imbalance), and handled the imbalance two ways: `class_weight="balanced"` during training, and tuning the decision threshold to 0.46 on the validation set rather than defaulting to 0.5.

**Execute.** I evaluated the tuned model once on the held-out test set, data it never saw during training or threshold selection.

## Results

The champion is the tuned random forest. On the held-out test set:

| Metric | Employees who left | Employees who stayed |
|---|---|---|
| Precision | 0.97 | 0.99 |
| Recall | 0.93 | 0.99 |
| F1 | 0.95 | 0.99 |

Overall accuracy was 0.98. The number I care about most is recall on the employees who left: 0.93 means the model catches 93% of the people who actually leave, and its 0.97 precision means when it raises a flag, it's almost always right. For a tool meant to trigger a retention conversation, those are the two that matter, since a missed leaver is a person out the door and a false alarm is only a check-in.

The feature importances line up with the EDA and give HR something concrete to act on:

| Feature | Importance |
|---|---|
| Satisfaction level | 0.30 |
| Tenure (years at company) | 0.24 |
| Number of projects | 0.16 |
| Average monthly hours | 0.14 |
| Last evaluation score | 0.13 |

Everything after those five (department, salary, work accidents, promotions) contributed almost nothing by comparison.

## What it means for HR

Satisfaction is the single biggest predictor of who leaves, so it's also the biggest lever. My recommendations:

Use the model as an early-warning system. Score employees, and route the ones it flags to a manager for a real conversation rather than an automated action. It's a way to surface risk for a human to follow up on, not a decision-maker.

Dig into what drives satisfaction, and act on it. Since satisfaction outranks pay, department, and evaluation scores as a predictor, that's where the investigation should start. Ask employees, especially those scoring low, what would actually change how they feel about the work, and build a satisfaction program around what comes back.

Watch tenure and workload. Tenure was the second-strongest signal and project count and monthly hours weren't far behind, which points at overwork and stalled careers as things worth examining alongside satisfaction.

Two honest caveats. Satisfaction is self-reported, so it's only as good as the survey. And the data is a single snapshot, so the model predicts on the patterns present when it was collected, not on anything that has shifted since.

## Repository structure

```
salifort-motors-employee-turnover/
├── README.md
├── Activity_ Course 7 Salifort Motors project lab.ipynb   # full analysis
├── HR_capstone_dataset.csv
└── .gitignore
```

## Tools

Python (pandas, NumPy, scikit-learn, category_encoders, matplotlib, seaborn), in Jupyter.

## About

I'm Faith Palombi, a high school math teacher moving into data and tech, with an earlier background in full-stack software engineering and physics research. This is one of several projects in my data analytics portfolio.
