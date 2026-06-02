
# Swim Sight 3D AI Server

Placeholder FastAPI server for Swim Sight 3D.

## Endpoints

GET /health  

Returns server health.

POST /process-video  

Receives a signed video URL from Base44, downloads the video temporarily, generates placeholder AI findings, and sends the result back to Base44 using the callback URL.

## Environment Variables

AI_WEBHOOK_SECRET  

Must match the Base44 AI_WEBHOOK_SECRET.

PORT  

Optional. Defaults to platform port.

## Run Locally

pip install -r requirements.txt

python -m uvicorn main:app --port 8000

