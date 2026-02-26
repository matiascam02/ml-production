# How-To: Reproduce the Customer Churn Prediction Project

## EuroConnect Telecom -- End-to-End Reproduction Guide

This document provides a step-by-step guide for reproducing the entire Customer Churn
Prediction pipeline, from local development through cloud deployment. Follow the steps
in order; each section builds on the previous one.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Setup](#2-local-setup)
3. [Running the EDA Notebook](#3-running-the-eda-notebook)
4. [Running Training Locally](#4-running-training-locally)
5. [Running Inference Locally](#5-running-inference-locally)
6. [Building Docker Images](#6-building-docker-images)
7. [Pushing to Artifact Registry](#7-pushing-to-artifact-registry)
8. [Creating Cloud Run Jobs](#8-creating-cloud-run-jobs)
9. [Deploying Cloud Workflows](#9-deploying-cloud-workflows)
10. [Setting Up Cloud Scheduler](#10-setting-up-cloud-scheduler)
11. [Running the Visualization Notebook](#11-running-the-visualization-notebook)
12. [Monitoring and Troubleshooting](#12-monitoring-and-troubleshooting)

---

## Reference Information

| Item                | Value                                                                 |
|---------------------|-----------------------------------------------------------------------|
| GCP Project ID      | `mlip-485615`                                                         |
| GCS Bucket          | `gs://greencart-churn-regression` (europe-west3)                      |
| Artifact Registry   | `europe-west3-docker.pkg.dev/mlip-485615/churn-prediction`            |
| Service Account     | `858518814941-compute@developer.gserviceaccount.com`                  |
| Region              | `europe-west3`                                                        |
| Dataset             | Telco Customer Churn (7,043 rows, 20 features)                       |
| Models              | Logistic Regression (lr), Random Forest (rf)                          |

---

## 1. Prerequisites

Before starting, make sure you have the following installed and configured on your
machine.

### Software

- **Python 3.11** -- The project is built and tested against Python 3.11. Check with:
  ```bash
  python3 --version
  ```
- **Docker** -- Required for building container images. Check with:
  ```bash
  docker --version
  ```
- **Google Cloud CLI (gcloud)** -- Required for all GCP interactions. Install from
  https://cloud.google.com/sdk/docs/install and check with:
  ```bash
  gcloud --version
  ```
- **Git** -- For cloning the repository.

### GCP Access

1. You need access to the GCP project `mlip-485615`. Ask the project owner to grant
   you the following IAM roles:
   - `roles/storage.objectAdmin` (read/write GCS bucket)
   - `roles/run.admin` (manage Cloud Run jobs)
   - `roles/workflows.admin` (manage Cloud Workflows)
   - `roles/cloudscheduler.admin` (manage Cloud Scheduler)
   - `roles/artifactregistry.writer` (push Docker images)
   - `roles/cloudbuild.builds.editor` (trigger Cloud Build)

2. Authenticate with gcloud:
   ```bash
   gcloud auth login
   gcloud config set project mlip-485615
   gcloud config set compute/region europe-west3
   ```

3. Set up Application Default Credentials (required for Python scripts that use
   `google-cloud-storage`):
   ```bash
   gcloud auth application-default login
   ```

---

## 2. Local Setup

### Clone the Repository

```bash
git clone <your-repo-url> ml-production
cd ml-production
```

### Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

For full local development (includes Jupyter, visualization libraries, etc.):

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `pandas`, `numpy` -- data manipulation
- `matplotlib`, `seaborn` -- visualization
- `scikit-learn`, `joblib` -- machine learning
- `google-cloud-storage` -- GCS integration
- `kagglehub` -- dataset download from Kaggle
- `jupyter`, `ipykernel` -- notebook support

There is also a `requirements-docker.txt` with a minimal set of dependencies used
inside Docker containers (no Jupyter or visualization libraries):
- `pandas`, `numpy`, `scikit-learn`, `joblib`, `google-cloud-storage`

### Verify the Dataset

The main dataset should already be present at `data/telco_customer_churn.csv`
(7,043 rows). It is also uploaded to GCS at `gs://greencart-churn-regression/inputs/telco_customer_churn.csv`.

To verify the GCS copy:

```bash
gsutil ls gs://greencart-churn-regression/inputs/
```

You should see:

```
gs://greencart-churn-regression/inputs/telco_customer_churn.csv
gs://greencart-churn-regression/inputs/test_unseen_sample.csv
```

---

## 3. Running the EDA Notebook

The exploratory data analysis notebook examines the dataset, identifies patterns, and
documents key findings about customer churn.

### Start Jupyter

```bash
jupyter notebook
```

### Open and Run the Notebook

1. In the Jupyter interface, navigate to `notebooks/exploratory_data_analysis.ipynb`.
2. Run all cells from top to bottom (Cell > Run All).
3. The notebook reads from `data/telco_customer_churn.csv` (local file).

### What the EDA Covers

- Dataset shape and data types
- Missing value analysis (11 blank TotalCharges values for new customers)
- Churn rate distribution (~26.5%)
- Feature-by-feature analysis against churn
- Key findings:
  - Month-to-month contracts have 42.7% churn vs. 2.8% for two-year contracts
  - Tenure has a strong negative correlation with churn (-0.35)
  - Fiber optic internet customers churn at 41.9% vs. 18.9% for DSL
  - Churners pay approximately $15 more per month on average

---

## 4. Running Training Locally

The training script (`src/training.py`) reads data from GCS, trains one or both
models with GridSearchCV, evaluates them, and saves artifacts back to GCS.

### Prerequisites

- You must be authenticated with `gcloud auth application-default login` (see step 1).
- The input dataset must exist at `gs://greencart-churn-regression/inputs/telco_customer_churn.csv`.

### Train Both Models

```bash
python src/training.py --model both
```

### Train Only Logistic Regression

```bash
python src/training.py --model lr
```

### Train Only Random Forest

```bash
python src/training.py --model rf
```

### Use a Custom Input File

```bash
python src/training.py --model both --input inputs/telco_customer_churn.csv
```

The `--input` flag specifies the path within the GCS bucket (not a local path).

### What Happens During Training

1. Connects to GCS and downloads the dataset.
2. Cleans data: converts `TotalCharges` to numeric, fills blanks with 0, encodes
   `Churn` as 0/1, drops `customerID`, casts `SeniorCitizen` to string.
3. Splits data 60/20/20 (train/validation/test) with stratification.
4. Builds a `ColumnTransformer` preprocessor:
   - `StandardScaler` for numerical features (`tenure`, `MonthlyCharges`,
     `TotalCharges`)
   - `OneHotEncoder` (drop first) for 16 categorical features
5. Trains models using GridSearchCV with 5-fold CV optimizing F1-score:
   - **Logistic Regression**: searches over `C` (0.001 to 100), `penalty` (l1/l2),
     `solver` (liblinear), with `class_weight='balanced'`.
   - **Random Forest**: searches over `n_estimators` (100/200), `max_depth`
     (3/5/10/None), `min_samples_leaf` (5/10/20), with `class_weight='balanced'`.
6. Evaluates on validation and test sets (Accuracy, Precision, Recall, F1, ROC-AUC).
7. Uploads to GCS:
   - `artifacts/preprocessor.joblib`
   - `artifacts/logistic_regression.joblib` (if lr or both)
   - `artifacts/random_forest.joblib` (if rf or both)
   - `outputs/training_metrics.csv`

### Expected Output

```
Connecting to GCS...

--- Loading Data ---
Loaded 7043 rows, 21 columns from gs://greencart-churn-regression/inputs/telco_customer_churn.csv
Cleaned: 7043 rows, churn rate: 26.5%

--- Splitting Data ---
  Train:  4225 samples (26.5% churn)
  Val  :  1409 samples (26.5% churn)
  Test :  1409 samples (26.5% churn)

--- Preprocessing ---
  Features after encoding: 30

--- Training Logistic Regression ---
  Best params: {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}
  Best CV F1:  0.6201
  ...

--- Training Random Forest ---
  Best params: {'class_weight': 'balanced', 'max_depth': 10, ...}
  Best CV F1:  0.6180
  ...

--- Saving Artifacts to GCS ---
  Uploaded: gs://greencart-churn-regression/artifacts/preprocessor.joblib
  Uploaded: gs://greencart-churn-regression/artifacts/logistic_regression.joblib
  Uploaded: gs://greencart-churn-regression/artifacts/random_forest.joblib
  Uploaded: gs://greencart-churn-regression/outputs/training_metrics.csv

Done.
```

---

## 5. Running Inference Locally

The inference script (`src/inference.py`) loads a trained model and preprocessor from
GCS, runs predictions on new data, and uploads the results back to GCS.

### Prerequisites

- Training must have been run at least once so that model artifacts exist in GCS.
- You must be authenticated with `gcloud auth application-default login`.

### Run Inference with Logistic Regression on the Test Sample

```bash
python src/inference.py --model lr --input inputs/test_unseen_sample.csv
```

### Run Inference with Random Forest

```bash
python src/inference.py --model rf --input inputs/test_unseen_sample.csv
```

### Run Inference on the Full Dataset (with labels)

```bash
python src/inference.py --model lr --input inputs/telco_customer_churn.csv
```

When the input data contains a `Churn` column, the script will also print evaluation
metrics (Accuracy, F1-Score, ROC-AUC) in addition to predictions.

### Custom Output Path

```bash
python src/inference.py --model lr --input inputs/test_unseen_sample.csv --output outputs/my_predictions.csv
```

### What Happens During Inference

1. Connects to GCS and downloads the preprocessor and chosen model.
2. Downloads the input CSV from GCS.
3. Cleans data (same transformations as training).
4. Applies the preprocessor and generates predictions + probabilities.
5. If the input contains a `Churn` column, prints evaluation metrics.
6. Uploads predictions to `gs://greencart-churn-regression/outputs/predictions.csv`
   (or custom path).

### Expected Output

```
Connecting to GCS...

--- Loading Artifacts ---
  Loaded: gs://greencart-churn-regression/artifacts/preprocessor.joblib
  Loaded: gs://greencart-churn-regression/artifacts/logistic_regression.joblib
  Using model: Logistic Regression

--- Loading Data ---
  Loaded 200 rows, 20 columns from gs://greencart-churn-regression/inputs/test_unseen_sample.csv
  Preprocessed 200 samples, 30 features

--- Running Predictions ---
  Predicted churners: XX / 200 (XX.X%)

--- Saving Predictions ---
  Uploaded: gs://greencart-churn-regression/outputs/predictions.csv

Done.
```

---

## 6. Building Docker Images

There are two Docker images: one for training and one for inference. Both use
`python:3.11-slim` as the base image and install only the minimal dependencies from
`requirements-docker.txt`.

### Build the Training Image

```bash
docker build -f Dockerfile.training -t europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/training:latest .
```

To also tag with a date:

```bash
docker build -f Dockerfile.training \
  -t europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/training:latest \
  -t europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/training:$(date +%Y-%m-%d) \
  .
```

### Build the Inference Image

```bash
docker build -f Dockerfile.inference -t europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/inference:latest .
```

To also tag with a date:

```bash
docker build -f Dockerfile.inference \
  -t europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/inference:latest \
  -t europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/inference:$(date +%Y-%m-%d) \
  .
```

### Using Cloud Build (Alternative)

Instead of building locally, you can use Cloud Build which builds in the cloud and
pushes the image to Artifact Registry automatically:

```bash
gcloud builds submit --config cloudbuild-training.yaml .
gcloud builds submit --config cloudbuild-inference.yaml .
```

These Cloud Build configs (`cloudbuild-training.yaml` and `cloudbuild-inference.yaml`)
tag each image with both `latest` and the current date tag (e.g., `2026-02-26`).

### Test the Docker Images Locally

To test the training image locally (requires GCP credentials to be mounted):

```bash
docker run --rm \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
  europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/training:latest \
  --model both
```

To test the inference image locally:

```bash
docker run --rm \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
  europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/inference:latest \
  --model lr --input inputs/test_unseen_sample.csv
```

### Dockerfile Details

**Dockerfile.training:**
- Base image: `python:3.11-slim`
- Copies `requirements-docker.txt` and `src/training.py` into `/app`
- Entrypoint: `python training.py`
- Default command: `--model both`

**Dockerfile.inference:**
- Base image: `python:3.11-slim`
- Copies `requirements-docker.txt` and `src/inference.py` into `/app`
- Entrypoint: `python inference.py`
- Default command: `--model lr`

---

## 7. Pushing to Artifact Registry

### One-Time Setup: Create the Repository (if it does not exist)

```bash
gcloud artifacts repositories create churn-prediction \
  --repository-format=docker \
  --location=europe-west3 \
  --description="Churn prediction Docker images"
```

### Configure Docker to Authenticate with Artifact Registry

```bash
gcloud auth configure-docker europe-west3-docker.pkg.dev
```

This only needs to be done once per machine. It updates your Docker config to use
gcloud as a credential helper for the `europe-west3-docker.pkg.dev` registry.

### Push the Training Image

```bash
docker push europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/training:latest
docker push europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/training:$(date +%Y-%m-%d)
```

### Push the Inference Image

```bash
docker push europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/inference:latest
docker push europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/inference:$(date +%Y-%m-%d)
```

### Verify the Images in Artifact Registry

```bash
gcloud artifacts docker images list europe-west3-docker.pkg.dev/mlip-485615/churn-prediction
```

You should see both `training` and `inference` images with their tags listed.

> **Note:** If you used Cloud Build in step 6, the images are already pushed to
> Artifact Registry automatically. You can skip the manual push.

---

## 8. Creating Cloud Run Jobs

Cloud Run Jobs run containerized workloads to completion (as opposed to Cloud Run
Services which listen for HTTP requests). We create two jobs: one for training and one
for inference.

### Create the Training Job

```bash
gcloud run jobs create churn-training \
  --image=europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/training:latest \
  --region=europe-west3 \
  --task-timeout=3600 \
  --memory=2Gi \
  --cpu=2 \
  --max-retries=1 \
  --args="--model,both" \
  --service-account=858518814941-compute@developer.gserviceaccount.com
```

### Create the Inference Job

```bash
gcloud run jobs create churn-inference \
  --image=europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/inference:latest \
  --region=europe-west3 \
  --task-timeout=1800 \
  --memory=1Gi \
  --cpu=1 \
  --max-retries=1 \
  --args="--model,lr" \
  --service-account=858518814941-compute@developer.gserviceaccount.com
```

### Update an Existing Job (when the image changes)

```bash
gcloud run jobs update churn-training \
  --image=europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/training:latest \
  --region=europe-west3

gcloud run jobs update churn-inference \
  --image=europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/inference:latest \
  --region=europe-west3
```

### Execute a Job Manually

```bash
gcloud run jobs execute churn-training --region=europe-west3 --wait
gcloud run jobs execute churn-inference --region=europe-west3 --wait
```

The `--wait` flag makes the command block until the job completes, printing logs
in real time. Remove it to execute asynchronously.

### List Jobs

```bash
gcloud run jobs list --region=europe-west3
```

---

## 9. Deploying Cloud Workflows

Cloud Workflows orchestrate the Cloud Run Jobs. There are two workflow definitions
in the repository:

- `training_orchestrator.yaml` -- triggers the `churn-training` Cloud Run Job
- `inference_orchestrator.yaml` -- triggers the `churn-inference` Cloud Run Job

### Deploy the Training Workflow

```bash
gcloud workflows deploy training-pipeline \
  --source=training_orchestrator.yaml \
  --location=europe-west3 \
  --service-account=858518814941-compute@developer.gserviceaccount.com
```

### Deploy the Inference Workflow

```bash
gcloud workflows deploy inference-pipeline \
  --source=inference_orchestrator.yaml \
  --location=europe-west3 \
  --service-account=858518814941-compute@developer.gserviceaccount.com
```

### Execute a Workflow Manually

```bash
gcloud workflows run training-pipeline --location=europe-west3
gcloud workflows run inference-pipeline --location=europe-west3
```

### Describe a Workflow

```bash
gcloud workflows describe training-pipeline --location=europe-west3
gcloud workflows describe inference-pipeline --location=europe-west3
```

### How the Workflows Work

Each workflow YAML follows the same pattern:

1. **init** -- assigns variables for project ID, region, and job name.
2. **run_[training|inference]** -- calls the Cloud Run Jobs API
   (`googleapis.run.v1.namespaces.jobs.run`) to execute the job.
3. **check_[training|inference]** -- logs the execution name.
4. **return_result** -- returns a success status and execution name.

---

## 10. Setting Up Cloud Scheduler

Cloud Scheduler triggers the workflows on a cron schedule.

### Create the Training Schedule

Runs every 2 days at 2:00 AM UTC:

```bash
gcloud scheduler jobs create http training-schedule \
  --location=europe-west3 \
  --schedule="0 2 */2 * *" \
  --uri="https://workflowexecutions.googleapis.com/v1/projects/mlip-485615/locations/europe-west3/workflows/training-pipeline/executions" \
  --http-method=POST \
  --oauth-service-account-email=858518814941-compute@developer.gserviceaccount.com \
  --time-zone="UTC"
```

### Create the Inference Schedule

Runs daily at 6:00 AM UTC:

```bash
gcloud scheduler jobs create http inference-schedule \
  --location=europe-west3 \
  --schedule="0 6 * * *" \
  --uri="https://workflowexecutions.googleapis.com/v1/projects/mlip-485615/locations/europe-west3/workflows/inference-pipeline/executions" \
  --http-method=POST \
  --oauth-service-account-email=858518814941-compute@developer.gserviceaccount.com \
  --time-zone="UTC"
```

### Trigger a Schedule Manually (for testing)

```bash
gcloud scheduler jobs run training-schedule --location=europe-west3
gcloud scheduler jobs run inference-schedule --location=europe-west3
```

### List Scheduler Jobs

```bash
gcloud scheduler jobs list --location=europe-west3
```

### Update a Schedule

```bash
gcloud scheduler jobs update http training-schedule \
  --location=europe-west3 \
  --schedule="0 3 */2 * *"
```

### Pause and Resume a Schedule

```bash
gcloud scheduler jobs pause training-schedule --location=europe-west3
gcloud scheduler jobs resume training-schedule --location=europe-west3
```

### Schedule Summary

| Schedule Name       | Cron Expression   | Frequency                | Triggers               |
|---------------------|-------------------|--------------------------|------------------------|
| training-schedule   | `0 2 */2 * *`     | Every 2 days at 2 AM UTC | training-pipeline      |
| inference-schedule  | `0 6 * * *`       | Daily at 6 AM UTC        | inference-pipeline     |

---

## 11. Running the Visualization Notebook

The data visualization notebook generates charts and plots for project reporting.

### Start Jupyter

```bash
jupyter notebook
```

### Open and Run the Notebook

1. Navigate to `notebooks/data_visualizations.ipynb`.
2. Run all cells from top to bottom (Cell > Run All).
3. The notebook uses `matplotlib` and `seaborn` for plotting.

Make sure you have run the training pipeline at least once, as some visualizations
may rely on outputs stored in `data/` or metrics from GCS.

---

## 12. Monitoring and Troubleshooting

### Checking Cloud Run Job Executions

View the latest execution of a job:

```bash
gcloud run jobs executions list --job=churn-training --region=europe-west3 --limit=5
gcloud run jobs executions list --job=churn-inference --region=europe-west3 --limit=5
```

View logs for a specific execution:

```bash
gcloud run jobs executions describe <EXECUTION_NAME> --region=europe-west3
```

### Viewing Logs

View real-time logs for the training job:

```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=churn-training" \
  --project=mlip-485615 \
  --limit=50 \
  --format="table(timestamp, textPayload)"
```

View real-time logs for the inference job:

```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=churn-inference" \
  --project=mlip-485615 \
  --limit=50 \
  --format="table(timestamp, textPayload)"
```

### Viewing Workflow Executions

```bash
gcloud workflows executions list training-pipeline --location=europe-west3 --limit=5
gcloud workflows executions list inference-pipeline --location=europe-west3 --limit=5
```

View details of a specific execution:

```bash
gcloud workflows executions describe <EXECUTION_ID> \
  --workflow=training-pipeline \
  --location=europe-west3
```

### Checking GCS Outputs

Verify that training artifacts were uploaded:

```bash
gsutil ls -l gs://greencart-churn-regression/artifacts/
```

Verify that predictions were generated:

```bash
gsutil ls -l gs://greencart-churn-regression/outputs/
```

Download and inspect predictions locally:

```bash
gsutil cp gs://greencart-churn-regression/outputs/predictions.csv .
head -20 predictions.csv
```

Download and inspect training metrics:

```bash
gsutil cp gs://greencart-churn-regression/outputs/training_metrics.csv .
cat training_metrics.csv
```

### Common Issues and Fixes

**Issue: "Permission denied" when accessing GCS**
- Make sure you have run `gcloud auth application-default login`.
- Verify that the service account has `roles/storage.objectAdmin` on the bucket.
- For Cloud Run Jobs, ensure the job is configured with the correct service account
  (`858518814941-compute@developer.gserviceaccount.com`).

**Issue: "Artifact not found" during inference**
- Training must be run before inference. The inference script expects
  `artifacts/preprocessor.joblib` and the model file to exist in GCS.
- Re-run training: `python src/training.py --model both`

**Issue: Docker build fails**
- Make sure you are running the build from the project root directory (where the
  Dockerfiles are located).
- Check that `requirements-docker.txt` and the relevant `src/*.py` file exist.

**Issue: Cloud Run Job fails with OOM (Out of Memory)**
- Increase the memory allocation:
  ```bash
  gcloud run jobs update churn-training --memory=4Gi --region=europe-west3
  ```

**Issue: Cloud Scheduler not triggering**
- Check the job status: `gcloud scheduler jobs describe training-schedule --location=europe-west3`
- Make sure the job is not paused.
- Verify the service account has permission to invoke Workflows.

**Issue: "Image not found" when creating Cloud Run Job**
- Ensure the Docker image has been pushed to Artifact Registry.
- Verify with: `gcloud artifacts docker images list europe-west3-docker.pkg.dev/mlip-485615/churn-prediction`
- Check that the image tag matches exactly.

### GCP Console Links

- **Cloud Run Jobs:** https://console.cloud.google.com/run/jobs?project=mlip-485615
- **Cloud Workflows:** https://console.cloud.google.com/workflows?project=mlip-485615
- **Cloud Scheduler:** https://console.cloud.google.com/cloudscheduler?project=mlip-485615
- **Artifact Registry:** https://console.cloud.google.com/artifacts/docker/mlip-485615/europe-west3/churn-prediction?project=mlip-485615
- **Cloud Storage:** https://console.cloud.google.com/storage/browser/greencart-churn-regression?project=mlip-485615
- **Cloud Logging:** https://console.cloud.google.com/logs?project=mlip-485615

---

## Quick Reference: Full Deployment from Scratch

For convenience, here is the complete sequence of commands to deploy the entire project
from a fresh start (after prerequisites are met):

```bash
# 1. Clone and set up locally
git clone <your-repo-url> ml-production
cd ml-production
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Authenticate with GCP
gcloud auth login
gcloud config set project mlip-485615
gcloud config set compute/region europe-west3
gcloud auth application-default login
gcloud auth configure-docker europe-west3-docker.pkg.dev

# 3. Run training locally to verify everything works
python src/training.py --model both

# 4. Run inference locally to verify
python src/inference.py --model lr --input inputs/test_unseen_sample.csv

# 5. Create Artifact Registry repository (one-time)
gcloud artifacts repositories create churn-prediction \
  --repository-format=docker \
  --location=europe-west3 \
  --description="Churn prediction Docker images"

# 6. Build and push Docker images via Cloud Build
gcloud builds submit --config cloudbuild-training.yaml .
gcloud builds submit --config cloudbuild-inference.yaml .

# 7. Create Cloud Run Jobs
gcloud run jobs create churn-training \
  --image=europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/training:latest \
  --region=europe-west3 \
  --task-timeout=3600 \
  --memory=2Gi \
  --cpu=2 \
  --max-retries=1 \
  --args="--model,both" \
  --service-account=858518814941-compute@developer.gserviceaccount.com

gcloud run jobs create churn-inference \
  --image=europe-west3-docker.pkg.dev/mlip-485615/churn-prediction/inference:latest \
  --region=europe-west3 \
  --task-timeout=1800 \
  --memory=1Gi \
  --cpu=1 \
  --max-retries=1 \
  --args="--model,lr" \
  --service-account=858518814941-compute@developer.gserviceaccount.com

# 8. Deploy Cloud Workflows
gcloud workflows deploy training-pipeline \
  --source=training_orchestrator.yaml \
  --location=europe-west3 \
  --service-account=858518814941-compute@developer.gserviceaccount.com

gcloud workflows deploy inference-pipeline \
  --source=inference_orchestrator.yaml \
  --location=europe-west3 \
  --service-account=858518814941-compute@developer.gserviceaccount.com

# 9. Set up Cloud Scheduler
gcloud scheduler jobs create http training-schedule \
  --location=europe-west3 \
  --schedule="0 2 */2 * *" \
  --uri="https://workflowexecutions.googleapis.com/v1/projects/mlip-485615/locations/europe-west3/workflows/training-pipeline/executions" \
  --http-method=POST \
  --oauth-service-account-email=858518814941-compute@developer.gserviceaccount.com \
  --time-zone="UTC"

gcloud scheduler jobs create http inference-schedule \
  --location=europe-west3 \
  --schedule="0 6 * * *" \
  --uri="https://workflowexecutions.googleapis.com/v1/projects/mlip-485615/locations/europe-west3/workflows/inference-pipeline/executions" \
  --http-method=POST \
  --oauth-service-account-email=858518814941-compute@developer.gserviceaccount.com \
  --time-zone="UTC"

# 10. Verify by executing jobs manually
gcloud run jobs execute churn-training --region=europe-west3 --wait
gcloud run jobs execute churn-inference --region=europe-west3 --wait
```
