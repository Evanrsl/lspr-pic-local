## Variables

- PROJECT_ID = ecomdiv
- DOCKER_IMAGE_NAME = lspr-pic
- DOCKER_SERVICE_NAME = lspr-pic
- REGION = asia-southeast2
- TIMEZONE = "Asia/Jakarta"
- BUCKET-NAME = lspr-pic-assign

## Pre-requisite

1. Google Cloud Initialization

    ```bash
   gcloud init
   ```

2. Delete virtual environment Directory

## Steps to Re-run Commands


1. Build and Push Docker Image:

   Run again if you have made changes to your Flask app or the Dockerfile

   ```bash
   gcloud builds submit --tag gcr.io/ecomdiv/lspr-pic
   ```

2. Deploy Docker Image to Cloud Run:

   Run again if you have pushed a new Docker image.

   ```bash
   gcloud run deploy lspr-pic --image gcr.io/ecomdiv/lspr-pic --platform managed --region asia-southeast2 --allow-unauthenticated
   ```

3. Create Pub/Sub Topic:

   Run again only if you haven't created the Pub/Sub topic yet or if you need to create a new one.

   ```bash
   gcloud pubsub topics create train-lspr-pic
   ```

4. Create Cloud Scheduler Job:

   Run again only if you haven't set up the scheduler yet or if you need to update the schedule.

   ```bash
   gcloud scheduler jobs create pubsub train-model-job --schedule "0 0 * * *" --time-zone "Asia/Jakarta" --topic train-lspr-pic --message-body '{}'
   ```

5. Deploy Cloud Function:

   Run again if you have made changes to the training.py script or other code that the Cloud Function relies on.

   ```bash
   gcloud functions deploy train_model_function --runtime python39 --trigger-topic train-lspr-pic --entry-point main --source .

   ```
