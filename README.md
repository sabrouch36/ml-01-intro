# ml-01-intro

[![Workflow Guide](https://img.shields.io/badge/Pro--Guide-pro--analytics--02-green)](https://denisecase.github.io/pro-analytics-02/workflow-b-apply-example-project/)
[![Python 3.14](https://img.shields.io/badge/python-3.14%2B-blue?logo=python)](./pyproject.toml)
[![MIT](https://img.shields.io/badge/license-see%20LICENSE-yellow.svg)](./LICENSE)

> Professional Python project: characterizing machine learning.

## Project Description

This project focuses on learning to find good data problems in a dataset,
and learning when machine learning (ML) might be helpful.

We learn to characterize:

- supervised (when we pick a target to predict)
- unsupervised (no target, just exploring, e.g. clustering)

In this project, we pick a dataset and a target.

If the target is:

- a discrete category column, we know it is a classification problem
- a continuous numeric column, we know it is a regression problem.

Some numbers are actually categories, for example a rating of 1, 2, 3.
May be better characterized as a category / discrete variable.

## Example Notebook + Your Notebook

Keep the example notebook as it is.
Either copy it or use it to build a new notebook that ends in _yourname.
See [docs/your-files.md] for more.

Links:

- [ml_01_case.ipynb](notebooks/ml_01_case.ipynb)

## Working Files

You'll work with these areas:

- **data/raw** - raw data for exploration (only if you add a dataset)
- **docs/** - project narrative and documentation
- **src/mlstudio/** - the app is an example; run only (no need to modify)
- **notebooks/** - interactive analysis
- **pyproject.toml** - update authorship & links
- **zensical.toml** - update authorship & links

## Instructions (pro-analytics-02)

Follow the
[step-by-step workflow guide](https://denisecase.github.io/pro-analytics-02/workflow-b-apply-example-project/)
to complete:

1. Phase 1. **Start & Run**
2. Phase 2. **Change Authorship**
3. Phase 3. **Read & Understand**
4. Phase 4. **Modify**
5. Phase 5. **Apply** <mark>(optional for Module 1)</mark>

**Completing Phases 1-4 is the goal for Module 1.**
Phase 5 is optional in Module 1.
If your environment is working well and you still have some time, you might try it.

## Challenges

Challenges are expected.
Sometimes instructions may not quite match your operating system.
When issues occur, share screenshots, error messages, and details about what you tried.
Working through issues is part of implementing professional projects.

## Success

After completing Phase 1. **Start & Run**, you'll have your own GitHub project,
with the example notebook executed and committed,
and running the example module will print out:

```shell
========================
Executed successfully!
========================
```

A new file `project.log` will appear in the root project folder.

## Command Reference

<details>
<summary>Show command reference</summary>

### In a machine terminal (open in your `Repos` folder)

After you get a copy of this repo in your own GitHub account,
open a machine terminal in your `Repos` folder:

```shell
# Replace username with YOUR GitHub username.
git clone https://github.com/sabrouch36/ml-01-intro

cd ml-01-intro
code .
```

### In a VS Code terminal

These are listed for convenience.
For best results, follow the detailed instructions in
[pro-analytics-02 guide](https://denisecase.github.io/pro-analytics-02/).

```shell
uv self update
uv python pin 3.14
uv lock --upgrade
uv sync --extra dev --extra docs --upgrade

uvx pre-commit install
uvx pre-commit autoupdate

# git add all files and auto fix them as much as possible while working
git add -A
uvx pre-commit run --all-files
# repeat if changes were made
uvx pre-commit run --all-files

# run the example module to verify the environment (.venv/)
uv run python -m mlstudio.app_case

# run common chores: format, lint, run checks and tests...
uv run ruff format .
uv run ruff check . --fix
uv run python -m pyright
uv run python -m pytest
uv run python -m zensical build

# save progress after every major change (customize the commit message)
git add -A
git commit -m "update"
git push -u origin main
```

</details>

## Notes

- Use the **UP ARROW** and **DOWN ARROW** in the terminal to scroll through past commands.
- Use `CTRL+f` to find (and replace) text within a file.
- You do not need to add to or modify `tests/`. They are provided for example only.
- Many files are silent helpers. Explore as you like, but nothing is required.
- You do NOT need to understand everything; understanding builds naturally over time.

## Troubleshooting >>>

If you see something like this in your terminal: `>>>` or `...`
You accidentally started Python interactive mode.
It happens.
Press `Ctrl+c` (both keys together) or `Ctrl+Z` then `Enter` on Windows.

## Example Output (Can Remove this Section after You Verify)

```shell
| INFO | ML | Summarize workflow........
| INFO | ML | ========================
| INFO | ML | SUMMARY
| INFO | ML | ========================
| INFO | ML | Dataset: hours_scores_case
| INFO | ML | Original rows: 10
| INFO | ML | Clean rows: 10
| INFO | ML | Features: ['hours_studied', 'practice_quizzes', 'attendance_pct', 'sleep_hours', 'prior_score']
| INFO | ML | Target: score
| INFO | ML | ----- in a script, call plt.show() once at the end to display all charts -----
| INFO | ML | ----- in a script, CLOSE the chart windows with the close button to CONTINUE -----
| INFO | ML | Workflow complete
| INFO | ML | IMPORTANT: This script creates chart windows.
| INFO | ML | Close chart windows and terminate this process with CTRL+c as needed.
| INFO | ML | ========================
| INFO | ML | Executed successfully!
| INFO | ML | ========================
```

## Findings and Visuals

Take screenshots of your charts and provide them here with a discussion.
In Markdown, display a figure by using:
an exclamation mark immediately followed by square brackets containing a useful caption
immediately followed by parentheses containing the relative path to your figure.
Note: When you start typing the path with a dot (.) for "here, in this directory",
the IDE may help complete the path.

In your custom project, follow this example, but

- your figures and narrative should reflect your work,
- this `README.md` should include your commands, process, and visuals, and
- `docs/index.md` should include your narrative.

Remove unnecessary instructional comments in your custom files.

These are from the example app used to test the .venv/.
If possible, replace these to present interesting results from your custom project:

![Provide a Useful Caption](./docs/images/Figure_1.png)

![Provide a Useful Caption](./docs/images/Figure_2.png)

## Project Documentation

Additional project instructions, terms, and notes:

[docs/index.md](docs/index.md)

## Citation

[CITATION.cff](./CITATION.cff)

## License

[MIT](./LICENSE)

Phase 4: Make a Technical Modification

## Custom Technical Modification

This project includes a custom supervised regression application that predicts
student scores from study-related features.

The custom implementation is located in:

- `src/mlstudio/app_sabri.py`

The custom dataset is located in:

- `data/raw/hours_scores_sabri.csv`

### Modification

I added a derived feature named `study_engagement`.

It combines the number of hours studied with the student's attendance percentage:

```python
study_engagement = hours_studied * attendance_pct / 100

This modification expands the model from five original features to six features:

hours_studied
practice_quizzes
attendance_pct
sleep_hours
prior_score
study_engagement

The derived feature is created both for the training dataset and for any new
case submitted to the trained model.

Machine Learning Workflow

The custom application performs the following steps:

Loads the student performance dataset.
Inspects the dataset structure.
Checks for missing values and duplicate rows.
Creates the derived study_engagement feature.
Splits the data into training and testing sets.
Trains a Linear Regression model.
Evaluates the model using MAE and R-squared.
Predicts the score for one new student case.
Creates and saves two visualizations.
Results

The latest successful execution produced the following results:

Dataset rows: 10
Missing values: 0
Duplicate rows: 0
Mean Absolute Error: 0.48
R-squared: 1.00
Predicted score for the new case: 83.4

The predicted student case used:

6.5 study hours
4 practice quizzes
92% attendance
7 hours of sleep
Previous score of 72
Calculated study engagement of 5.98

The results show that hours_studied had the strongest positive model
coefficient.

Because the dataset contains only 10 records and several related features,
the evaluation results should be interpreted cautiously. The high R-squared
value may indicate that the model is overfitting this small educational dataset.

Visualizations
Hours Studied vs Student Score

Linear Regression Model Coefficients

Run the Custom Application

Run the following command from the project root folder:

uv run python -m mlstudio.app_sabri

Close the chart windows after reviewing them to allow the application to
finish successfully.
