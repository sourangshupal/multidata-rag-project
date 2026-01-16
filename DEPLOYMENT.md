# AWS Lambda Deployment Guide

Complete step-by-step guide for deploying the Multi-Source RAG + Text-to-SQL application to AWS Lambda with GitHub Actions CI/CD.

---

## Table of Contents

1. [Overview](#overview)
2. [What We Built](#what-we-built)
3. [Prerequisites](#prerequisites)
4. [Step 1: Verify Local Setup](#step-1-verify-local-setup)
5. [Step 2: Configure AWS CLI](#step-2-configure-aws-cli)
6. [Step 3: Create AWS Resources](#step-3-create-aws-resources)
7. [Step 4: Configure GitHub](#step-4-configure-github)
8. [Step 5: Deploy to Lambda](#step-5-deploy-to-lambda)
9. [Step 6: Verify Deployment](#step-6-verify-deployment)
10. [Monitoring & Logs](#monitoring--logs)
11. [Updating the Application](#updating-the-application)
12. [Rollback Procedure](#rollback-procedure)
13. [Cost Breakdown](#cost-breakdown)
14. [Troubleshooting](#troubleshooting)
15. [FAQ](#faq)

---

## Overview

This guide will walk you through deploying your FastAPI application to AWS Lambda as a serverless containerized function. The deployment pipeline automatically builds, tests, and deploys your code whenever you push to the `main` branch.

**Architecture:**
```
GitHub Repository (push to main)
    ↓
GitHub Actions (CI/CD)
    ↓
AWS ECR (Container Registry)
    ↓
AWS Lambda (Serverless Compute)
    ↓
AWS API Gateway (HTTP Endpoint)
    ↓
Your Application is Live!
```

---

## What We Built

### Files Created

**Lambda-Specific Files:**
1. **`Dockerfile.lambda`** - Optimized Docker image for Lambda (without OCR)
2. **`Dockerfile.lambda.with-tesseract`** - Full Docker image with Tesseract OCR support
3. **`lambda_handler.py`** - Lambda entry point that wraps FastAPI with Mangum adapter

**CI/CD Workflows:**
4. **`.github/workflows/deploy.yml`** - Automatic deployment on push to main
5. **`.github/workflows/test.yml`** - Automatic testing on pull requests

**Deployment Documentation:**
6. **`DEPLOYMENT_FIXES.md`** - Complete troubleshooting guide for Lambda deployment
7. **`CROSS_PLATFORM_BUILD.md`** - Building Docker images on Windows/Mac/Linux
8. **`TEAM_SETUP.md`** - Setup guide for all team members

### Files Modified

1. **`app/config.py`** - Added Lambda storage paths and environment detection
2. **`app/main.py`** - Added `initialize_services()` function for Lambda compatibility
3. **`app/logging_config.py`** - Added CloudWatch-compatible logging for Lambda

### Key Features

- ✅ **Automatic Deployment**: Push to `main` → Auto-deploy to Lambda
- ✅ **ARM64 Architecture**: 20% cheaper than x86_64, better performance
- ✅ **Service Initialization**: Manual initialization for Mangum compatibility
- ✅ **Cross-Platform Builds**: Build on Windows, Mac (Intel/ARM), or Linux
- ✅ **Ephemeral Storage**: Uses Lambda's `/tmp` directory (10GB)
- ✅ **CloudWatch Logging**: All logs automatically sent to CloudWatch
- ✅ **No Infrastructure as Code**: Simple AWS CLI commands, no Terraform
- ✅ **File Permissions**: Properly configured for Lambda runtime access
- ✅ **API Gateway Integration**: HTTP API v2 with /prod base path
- ✅ **Cost Effective**: ~$40-70/month for 100K requests (ARM64)

---

## Prerequisites

### 1. Software Requirements

Make sure you have these installed on your local machine:

- **AWS CLI** (version 2.x)
- **Git**
- **Docker** (for local testing)
- **Python 3.12** (for local testing)

#### Install AWS CLI

**macOS:**
```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
```

**Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

**Windows:**
Download and run: https://awscli.amazonaws.com/AWSCLIV2.msi

**Verify installation:**
```bash
aws --version
# Should output: aws-cli/2.x.x Python/3.x.x ...
```

### 2. AWS Account Requirements

You'll need:
- An AWS account with admin access (or permissions for ECR, Lambda, IAM, API Gateway)
- AWS Access Key ID and Secret Access Key
- Billing enabled (Lambda free tier: 1M free requests/month)

### 3. API Keys & Services

Your application needs these external services:
- **OpenAI API Key** - For embeddings and LLM (https://platform.openai.com/api-keys)
- **Pinecone API Key** - For vector storage (https://www.pinecone.io/)
- **Supabase/PostgreSQL URL** - For SQL database (https://supabase.com/)
- **OPIK API Key** (Optional) - For monitoring (https://www.comet.com/opik)

---

## Step 1: Verify Local Setup

Before deploying to AWS, let's verify everything works locally.

### 1.1 Check Repository Structure

```bash
cd /Users/sourangshupal/Downloads/multidata-rag-project

# Verify key files exist
ls -la Dockerfile.lambda
ls -la lambda_handler.py
ls -la .github/workflows/deploy.yml
ls -la .github/workflows/test.yml
```

You should see all 4 files listed.

### 1.2 Test Lambda Docker Image Locally (Optional)

Build and test the Lambda container:

```bash
# Build the Lambda image for ARM64 (recommended - 20% cheaper)
docker build --platform linux/arm64 -f Dockerfile.lambda -t rag-lambda:test .

# Alternative: Build for x86_64 (if needed)
# docker build --platform linux/amd64 -f Dockerfile.lambda -t rag-lambda:test .

# This will take 5-10 minutes the first time
# Watch for any errors during the build
```

**Note**: Using `--platform linux/arm64` ensures your image matches Lambda's ARM64 architecture. Cross-platform builds work on any host machine (Windows/Mac/Linux).

If the build succeeds, your Lambda image is ready!

**For detailed build instructions**, see `CROSS_PLATFORM_BUILD.md` in the project root.

---

## Step 2: Configure AWS CLI

### 2.1 Get AWS Credentials

1. Log into AWS Console: https://console.aws.amazon.com/
2. Go to **IAM** → **Users** → Click your username
3. Click **Security credentials** tab
4. Click **Create access key**
5. Select "Command Line Interface (CLI)"
6. Click "Next" → "Create access key"
7. **Download the CSV** or copy:
   - Access Key ID (starts with `AKIA...`)
   - Secret Access Key (long random string)

⚠️ **IMPORTANT**: Keep these credentials secret! Never commit them to Git.

### 2.2 Configure AWS CLI

Run this command and enter your credentials:

```bash
aws configure
```

**Prompts:**
```
AWS Access Key ID [None]: AKIA...YOUR_KEY_HERE
AWS Secret Access Key [None]: YOUR_SECRET_KEY_HERE
Default region name [None]: us-east-1
Default output format [None]: json
```

**Explanation:**
- **Access Key ID**: Your AWS identity
- **Secret Access Key**: Your AWS password (never share this!)
- **Region**: `us-east-1` (US East - Virginia) - cheapest and most features
- **Output format**: `json` - makes output easy to read

### 2.3 Verify AWS Access

Test that AWS CLI is configured correctly:

```bash
aws sts get-caller-identity
```

**Expected output:**
```json
{
    "UserId": "AIDACKCEVSQ6C2EXAMPLE",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/YourName"
}
```

**If you see an error**: Your credentials are incorrect. Run `aws configure` again.

---

## Step 3: Create AWS Resources

Now we'll create all the AWS resources needed for deployment. **This is a one-time setup.**

### 3.1 Get Your AWS Account ID

First, save your AWS account ID to an environment variable:

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Your AWS Account ID: $AWS_ACCOUNT_ID"
```

**Explanation**: This command gets your 12-digit AWS account ID and stores it in a variable we'll use later.

---

### 3.2 Create ECR Repository

**What is ECR?** Amazon Elastic Container Registry - it's like Docker Hub, but for AWS. This stores your Docker images.

```bash
aws ecr create-repository \
  --repository-name rag-text-to-sql \
  --region us-east-1
```

**Expected output:**
```json
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:123456789012:repository/rag-text-to-sql",
        "registryId": "123456789012",
        "repositoryName": "rag-text-to-sql",
        "repositoryUri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/rag-text-to-sql"
    }
}
```

**Save the `repositoryUri`** - you'll use this later!

**If you see an error** "RepositoryAlreadyExistsException": The repository already exists, that's fine! Continue to the next step.

---

### 3.3 Create Lambda IAM Role

**What is IAM?** Identity and Access Management - it controls permissions in AWS.

**What is this role for?** Lambda needs permission to write logs to CloudWatch.

#### Step 3.3.1: Create Trust Policy File

This tells AWS that Lambda is allowed to use this role:

```bash
cat > trust-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
```

**Explanation**: This JSON file says "Lambda service is allowed to assume this role."

#### Step 3.3.2: Create the IAM Role

```bash
aws iam create-role \
  --role-name rag-lambda-execution-role \
  --assume-role-policy-document file://trust-policy.json
```

**Expected output:**
```json
{
    "Role": {
        "Path": "/",
        "RoleName": "rag-lambda-execution-role",
        "RoleId": "AROAEXAMPLEROLEID",
        "Arn": "arn:aws:iam::123456789012:role/rag-lambda-execution-role",
        "CreateDate": "2026-01-14T10:00:00Z"
    }
}
```

**If you see an error** "EntityAlreadyExists": The role already exists, that's fine!

#### Step 3.3.3: Attach Basic Lambda Execution Policy

This gives Lambda permission to write logs:

```bash
aws iam attach-role-policy \
  --role-name rag-lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

**No output means success!**

**Explanation**: The `AWSLambdaBasicExecutionRole` policy allows Lambda to create CloudWatch log groups and write logs.

---

### 3.4 Create Lambda Function

**What is Lambda?** AWS Lambda runs your code without managing servers. You pay only when your code runs.

#### Step 3.4.1: Get ECR Repository URI

```bash
export ECR_URI=$(aws ecr describe-repositories \
  --repository-names rag-text-to-sql \
  --query 'repositories[0].repositoryUri' \
  --output text)

echo "ECR URI: $ECR_URI"
```

**Expected output**: `123456789012.dkr.ecr.us-east-1.amazonaws.com/rag-text-to-sql`

#### Step 3.4.2: Build and Push Initial Docker Image

Before creating the Lambda function, we need to push an initial image to ECR:

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_URI

# Build the Lambda image for ARM64 (recommended)
docker build --platform linux/arm64 -f Dockerfile.lambda -t $ECR_URI:latest .

# Tag for ARM64
docker tag $ECR_URI:latest $ECR_URI:arm64

# Push both tags to ECR
docker push $ECR_URI:latest
docker push $ECR_URI:arm64
```

**This will take 5-10 minutes.** You'll see Docker building and pushing layers.

**Important**: We're building for ARM64 architecture because:
- 20% cheaper Lambda costs than x86_64
- Better price/performance ratio
- Works on all platforms with Docker's cross-platform build support

#### Step 3.4.3: Create the Lambda Function

```bash
aws lambda create-function \
  --function-name rag-text-to-sql \
  --package-type Image \
  --code ImageUri=${ECR_URI}:arm64 \
  --role arn:aws:iam::${AWS_ACCOUNT_ID}:role/rag-lambda-execution-role \
  --architectures arm64 \
  --timeout 900 \
  --memory-size 8192 \
  --ephemeral-storage Size=10240 \
  --region us-east-1
```

**Configuration Explained:**
- `--function-name`: Name of your Lambda function
- `--package-type Image`: We're using a Docker container (not a ZIP file)
- `--code ImageUri`: Points to the ARM64 Docker image in ECR
- `--role`: The IAM role we created (gives Lambda permissions)
- `--architectures arm64`: Uses ARM64 architecture (20% cheaper, better performance)
- `--timeout 900`: Maximum execution time (900 seconds = 15 minutes)
- `--memory-size 8192`: RAM allocated (8 GB - needed for docling/unstructured)
- `--ephemeral-storage`: Temporary disk space (10 GB in `/tmp`)

**Expected output:**
```json
{
    "FunctionName": "rag-text-to-sql",
    "FunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:rag-text-to-sql",
    "Role": "arn:aws:iam::123456789012:role/rag-lambda-execution-role",
    "Runtime": null,
    "MemorySize": 8192,
    "Timeout": 900,
    "State": "Pending"
}
```

**Wait for the function to be active:**
```bash
aws lambda wait function-active --function-name rag-text-to-sql
echo "Lambda function is now active!"
```

#### Step 3.4.4: Add Environment Variables

Your Lambda function needs API keys to work:

```bash
aws lambda update-function-configuration \
  --function-name rag-text-to-sql \
  --environment Variables="{
    ENVIRONMENT=production,
    OPENAI_API_KEY=your-openai-key-here,
    PINECONE_API_KEY=your-pinecone-key-here,
    DATABASE_URL=your-database-url-here,
    OPIK_API_KEY=your-opik-key-optional
  }"
```

**⚠️ REPLACE THE PLACEHOLDER VALUES:**
- `your-openai-key-here` → Your actual OpenAI API key (starts with `sk-`)
- `your-pinecone-key-here` → Your actual Pinecone API key
- `your-database-url-here` → Your PostgreSQL connection string (e.g., `postgresql://user:pass@host:5432/db`)
- `your-opik-key-optional` → Your OPIK key (or remove this if not using OPIK)

**Example:**
```bash
aws lambda update-function-configuration \
  --function-name rag-text-to-sql \
  --environment Variables="{
    ENVIRONMENT=production,
    OPENAI_API_KEY=sk-proj-abc123xyz...,
    PINECONE_API_KEY=abc123-456-789...,
    DATABASE_URL=postgresql://postgres:mypass@db.supabase.co:5432/postgres,
    OPIK_API_KEY=opik_abc123
  }"
```

**Wait for configuration update:**
```bash
aws lambda wait function-updated --function-name rag-text-to-sql
echo "Environment variables configured!"
```

---

### 3.5 Create HTTP API Gateway

**What is API Gateway?** It creates a public HTTPS endpoint that triggers your Lambda function.

#### Step 3.5.1: Create the HTTP API

```bash
API_ID=$(aws apigatewayv2 create-api \
  --name rag-api \
  --protocol-type HTTP \
  --query 'ApiId' \
  --output text)

echo "API ID: $API_ID"
```

**Example output**: `abc123xyz`

**Save this API_ID** - you'll need it!

#### Step 3.5.2: Create Lambda Integration

This connects API Gateway to your Lambda function:

```bash
INTEGRATION_ID=$(aws apigatewayv2 create-integration \
  --api-id $API_ID \
  --integration-type AWS_PROXY \
  --integration-uri arn:aws:lambda:us-east-1:${AWS_ACCOUNT_ID}:function/rag-text-to-sql \
  --payload-format-version 2.0 \
  --query 'IntegrationId' \
  --output text)


aws apigatewayv2 create-integration --api-id 9972ofec33 --integration-type AWS_PROXY --integration-uri arn:aws:lambda:us-east-1:120816008310:function:rag-text-to-sql --payload-format-version 2.0 --query 'IntegrationId' --output text

echo "Integration ID: $INTEGRATION_ID"
```

**Explanation:**
- `AWS_PROXY`: API Gateway forwards all requests directly to Lambda
- `payload-format-version 2.0`: Uses latest payload format (better for HTTP APIs)

#### Step 3.5.3: Create Routes

Routes tell API Gateway which requests to send to Lambda:

```bash
# Catch-all route (matches any path with subpaths)
aws apigatewayv2 create-route \
  --api-id $API_ID \
  --route-key 'ANY /{proxy+}' \
  --target integrations/$INTEGRATION_ID

# Root route (matches the base path)
aws apigatewayv2 create-route \
  --api-id $API_ID \
  --route-key 'ANY /' \
  --target integrations/$INTEGRATION_ID
```

**Explanation:**
- `ANY` = matches all HTTP methods (GET, POST, PUT, DELETE, etc.)
- `/{proxy+}` = matches any path (e.g., `/health`, `/upload`, `/query`)
- `/` = matches the root path exactly

#### Step 3.5.4: Create Deployment and Stage

```bash
# Create a deployment (activates the API)
aws apigatewayv2 create-deployment --api-id $API_ID

# Create a production stage
aws apigatewayv2 create-stage \
  --api-id $API_ID \
  --stage-name prod \
  --auto-deploy
```

**Explanation:**
- **Deployment**: Publishes your API changes
- **Stage**: A version of your API (e.g., dev, staging, prod)
- **auto-deploy**: Automatically deploys changes when you update the API

#### Step 3.5.5: Grant API Gateway Permission to Invoke Lambda

This is crucial! Without this, API Gateway can't trigger your Lambda:

```bash
aws lambda add-permission \
  --function-name rag-text-to-sql \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:us-east-1:${AWS_ACCOUNT_ID}:${API_ID}/*/*"
```

**Expected output:**
```json
{
    "Statement": "{\"Sid\":\"apigateway-invoke\",\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"apigateway.amazonaws.com\"},\"Action\":\"lambda:InvokeFunction\",\"Resource\":\"arn:aws:lambda:us-east-1:123456789012:function:rag-text-to-sql\",\"Condition\":{\"ArnLike\":{\"AWS:SourceArn\":\"arn:aws:execute-api:us-east-1:123456789012:abc123xyz/*/*\"}}}"
}
```

#### Step 3.5.6: Get Your API Endpoint URL

```bash
API_URL="https://${API_ID}.execute-api.us-east-1.amazonaws.com"
echo "========================================="
echo "🎉 Your API is now live!"
echo "========================================="
echo "API Endpoint: $API_URL"
echo ""
echo "Test it with:"
echo "curl $API_URL/health"
echo "========================================="
```

**Save this URL!** You'll use it for testing and GitHub configuration.

**Example URL**: `https://abc123xyz.execute-api.us-east-1.amazonaws.com`

---

### 3.6 Test Your API

Let's verify everything works:

```bash
# Test the health endpoint
curl $API_URL/health

# Test the info endpoint
curl $API_URL/info
```

**Expected response for `/health`:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-14T10:00:00",
  "services": {
    "embedding_service": true,
    "vector_service": true,
    "rag_service": true,
    "sql_service": true
  }
}
```

**Expected response for `/info`:**
```json
{
  "app_name": "Multi-Source RAG + Text-to-SQL",
  "version": "0.1.0",
  "environment": "production"
}
```

**If you get an error:**
- Wait 30 seconds and try again (Lambda cold start)
- Check CloudWatch logs (see [Monitoring section](#monitoring--logs))
- Verify environment variables are set correctly

---

## Step 4: Configure GitHub

Now let's set up GitHub to automatically deploy to Lambda when you push code.

### 4.1 Create GitHub Secrets

Secrets are encrypted environment variables used by GitHub Actions.

1. Go to your GitHub repository
2. Click **Settings** (top navigation)
3. Click **Secrets and variables** → **Actions** (left sidebar)
4. Click **New repository secret**

**Add these 4 secrets:**

#### Secret 1: AWS_ACCOUNT_ID
- **Name**: `AWS_ACCOUNT_ID`
- **Value**: Your 12-digit AWS account ID (from Step 2.3)
- Click "Add secret"

#### Secret 2: AWS_ACCESS_KEY_ID
- **Name**: `AWS_ACCESS_KEY_ID`
- **Value**: Your AWS access key (from Step 2.1)
- Click "Add secret"

#### Secret 3: AWS_SECRET_ACCESS_KEY
- **Name**: `AWS_SECRET_ACCESS_KEY`
- **Value**: Your AWS secret key (from Step 2.1)
- Click "Add secret"

#### Secret 4: API_GATEWAY_URL
- **Name**: `API_GATEWAY_URL`
- **Value**: Your API Gateway URL (from Step 3.5.6)
- Example: `https://abc123xyz.execute-api.us-east-1.amazonaws.com`
- Click "Add secret"

**Verify all 4 secrets are added:**
You should see them listed on the Actions secrets page.

---

## Step 5: Deploy to Lambda

Now everything is configured! Let's deploy.

### 5.1 Commit and Push Your Changes

```bash
# Make sure you're in the project directory
cd /Users/sourangshupal/Downloads/multidata-rag-project

# Check what files changed
git status

# Add all changes
git add .

# Commit with a descriptive message
git commit -m "Add Lambda CI/CD pipeline deployment"

# Push to main branch (this triggers deployment!)
git push origin main
```

### 5.2 Watch GitHub Actions

1. Go to your GitHub repository
2. Click **Actions** tab (top navigation)
3. You should see a workflow running: "Deploy to AWS Lambda"
4. Click on it to see the live progress

**Workflow steps:**
1. ✓ Checkout code
2. ✓ Configure AWS credentials
3. ✓ Login to Amazon ECR
4. ✓ Build, tag, and push image to ECR (takes 5-10 minutes)
5. ✓ Update Lambda function
6. ✓ Test deployment

**Total time: ~10-15 minutes for first deployment**

### 5.3 Deployment Success

When the workflow completes successfully, you'll see:
- ✅ All steps with green checkmarks
- "✅ Deployment successful!" message in logs

Your application is now live on AWS Lambda! 🎉

---

## Step 6: Verify Deployment

Let's make sure everything works correctly.

### 6.1 Test API Endpoints

```bash
# Set your API URL (replace with your actual URL from Step 3.5.6)
API_URL="https://your-api-id.execute-api.us-east-1.amazonaws.com"

# Test health endpoint
echo "Testing /health..."
curl $API_URL/health | jq

# Test info endpoint
echo "Testing /info..."
curl $API_URL/info | jq

# Test stats endpoint
echo "Testing /stats..."
curl $API_URL/stats | jq
```

**Note**: Install `jq` for pretty JSON output: `brew install jq` (macOS) or `sudo apt install jq` (Linux)

### 6.2 Test Document Upload

Create a test PDF file:

```bash
echo "This is a test document for RAG." > test.txt
# If you have a PDF, use that instead
```

Upload the document:

```bash
curl -X POST $API_URL/upload \
  -F "file=@test.txt" | jq
```

**Expected response:**
```json
{
  "status": "success",
  "filename": "test.txt",
  "doc_id": "abc123...",
  "chunks_created": 1,
  "cache_hit": false
}
```

### 6.3 Test Query

Query the uploaded document:

```bash
curl -X POST $API_URL/query/documents \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "top_k": 3
  }' | jq
```

**Expected response:**
```json
{
  "status": "success",
  "answer": "This document is a test document for RAG...",
  "sources": [...]
}
```

### 6.4 Test SQL Query

```bash
curl -X POST $API_URL/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many customers are in the database?",
    "auto_approve_sql": true
  }' | jq
```

**Expected response:**
```json
{
  "query_type": "SQL",
  "sql_generated": "SELECT COUNT(*) FROM customers",
  "results": [...],
  "execution_time_ms": 123
}
```

---

## Monitoring & Logs

### View Lambda Logs

**Real-time logs:**
```bash
aws logs tail /aws/lambda/rag-text-to-sql --follow
```

**Logs from last hour:**
```bash
aws logs tail /aws/lambda/rag-text-to-sql --since 1h
```

**Search for errors:**
```bash
aws logs tail /aws/lambda/rag-text-to-sql --filter-pattern "ERROR"
```

### View Lambda Metrics

**Get function info:**
```bash
aws lambda get-function --function-name rag-text-to-sql
```

**Check invocations (last 24 hours):**
```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --dimensions Name=FunctionName,Value=rag-text-to-sql \
  --start-time $(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 3600 \
  --statistics Sum
```

**Check error rate:**
```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Errors \
  --dimensions Name=FunctionName,Value=rag-text-to-sql \
  --start-time $(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 3600 \
  --statistics Sum
```

### CloudWatch Dashboard (Optional)

Create a dashboard in AWS Console:
1. Go to CloudWatch → Dashboards
2. Create new dashboard: "rag-text-to-sql-dashboard"
3. Add widgets:
   - Lambda Invocations
   - Lambda Errors
   - Lambda Duration
   - API Gateway 4xx/5xx Errors

---

## Updating the Application

### Automatic Updates (Recommended)

Simply push to the `main` branch:

```bash
# Make your code changes
vim app/main.py

# Commit and push
git add .
git commit -m "Update feature X"
git push origin main

# GitHub Actions automatically:
# 1. Builds new Docker image
# 2. Pushes to ECR
# 3. Updates Lambda function
# 4. Runs smoke tests
```

### Manual Update (If Needed)

If you need to update Lambda manually:

```bash
# Build new image for ARM64
docker build --platform linux/arm64 -f Dockerfile.lambda -t $ECR_URI:manual-update .

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_URI

# Push to ECR
docker push $ECR_URI:manual-update

# Update Lambda (IMPORTANT: Include --architectures flag)
aws lambda update-function-code \
  --function-name rag-text-to-sql \
  --image-uri $ECR_URI:manual-update \
  --architectures arm64

# Wait for update
aws lambda wait function-updated --function-name rag-text-to-sql
```

**Important**: The `--architectures` flag **MUST** be used with `update-function-code`, not with `update-function-configuration`. This is an AWS CLI requirement.

### Update Environment Variables

To update API keys or other environment variables:

```bash
aws lambda update-function-configuration \
  --function-name rag-text-to-sql \
  --environment Variables="{
    ENVIRONMENT=production,
    OPENAI_API_KEY=new-key-here,
    PINECONE_API_KEY=new-key-here,
    DATABASE_URL=new-url-here
  }"
```

---

## Rollback Procedure

### View Previous Versions

```bash
# List all ECR images (sorted by push date)
aws ecr describe-images \
  --repository-name rag-text-to-sql \
  --query 'sort_by(imageDetails,& imagePushedAt)[*].[imageTags[0], imagePushedAt]' \
  --output table
```

### Rollback to Previous Version

```bash
# Get the second-most-recent image tag
PREVIOUS_TAG=$(aws ecr describe-images \
  --repository-name rag-text-to-sql \
  --query 'sort_by(imageDetails,& imagePushedAt)[-2].imageTags[0]' \
  --output text)

echo "Rolling back to: $PREVIOUS_TAG"

# Update Lambda to previous version
aws lambda update-function-code \
  --function-name rag-text-to-sql \
  --image-uri ${ECR_URI}:${PREVIOUS_TAG}

# Wait for update
aws lambda wait function-updated --function-name rag-text-to-sql

echo "✅ Rollback complete!"
```

### Verify Rollback

```bash
# Test the API
curl $API_URL/health
curl $API_URL/info

# Check logs for errors
aws logs tail /aws/lambda/rag-text-to-sql --since 5m
```

---

## Cost Breakdown

### AWS Costs (Monthly Estimate)

Based on **100,000 requests/month**, **30-second average duration**, **8GB RAM**, **ARM64 architecture**:

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| **Lambda (ARM64)** | 8GB RAM, 100K invocations, 30s avg | **$40-65** |
| **API Gateway (HTTP)** | 1M requests | **$1.00** |
| **ECR** | 5GB storage (3 image versions) | **$0.50** |
| **CloudWatch Logs** | 10GB logs, 7-day retention | **$5.00** |
| **Data Transfer** | 10GB outbound | **$0.90** |
| **Total (AWS)** | | **~$47-72/month** |

**ARM64 Savings**: Using ARM64 instead of x86_64 saves ~$10-15/month (20% cheaper Lambda costs)

### External Services (Estimate)

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| **OpenAI** | 100K embeddings + 50K LLM calls | **$10-30** |
| **Pinecone** | Starter plan (100K vectors) | **$70-100** |
| **Supabase** | Free tier or Pro ($25/month) | **$0-25** |
| **Total (External)** | | **$80-155/month** |

### **Grand Total: $127-227/month**

**Previous (x86_64): $137-242/month**
**Savings with ARM64: ~$10-15/month (7-10% total cost reduction)**

### Cost Optimization Tips

1. **Use ARM64 architecture** (already configured ✓) - saves 20% on Lambda costs
2. **Use Lambda Reserved Concurrency** to limit max concurrent executions
3. **Reduce CloudWatch log retention** to 3 days instead of 7
4. **Use Pinecone Serverless** (pay per use) instead of dedicated pods
5. **Cache frequently accessed documents** to reduce OpenAI calls
6. **Set up CloudWatch alarms** to catch cost spikes early

---

## Troubleshooting

### Problem: GitHub Actions deployment fails

**Error**: "Unable to locate credentials"

**Solution**: Check GitHub secrets are configured correctly:
```bash
# In GitHub: Settings → Secrets → Actions
# Verify these exist:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_ACCOUNT_ID
# - API_GATEWAY_URL
```

---

### Problem: Lambda function times out

**Error**: "Task timed out after 900.00 seconds"

**Solution**: Large documents may exceed 15-minute timeout. Options:
1. Split document into smaller chunks
2. Process documents asynchronously (requires code changes)
3. Use Step Functions for long-running workflows

---

### Problem: API Gateway returns 502 Bad Gateway

**Error**: `{"message":"Internal server error"}`

**Solution**:
1. Check Lambda logs:
   ```bash
   aws logs tail /aws/lambda/rag-text-to-sql --since 10m
   ```
2. Common causes:
   - Lambda function crashed (check for Python errors in logs)
   - Out of memory (increase `--memory-size`)
   - Environment variables not set correctly

---

### Problem: API Gateway returns 404 Not Found

**Error**: "The API with ID xxx doesn't include a route with path /*"

**Common Causes:**
1. **Architecture mismatch**: Docker image built for wrong architecture (ARM64 vs x86_64)
2. **API Gateway integration**: Points to wrong Lambda function or account
3. **Mangum base path**: Missing API Gateway base path configuration

**Solutions:**

**1. Verify Lambda architecture matches Docker image:**
```bash
# Check Docker image architecture
docker inspect $ECR_URI:latest | grep Architecture

# Check Lambda function architecture
aws lambda get-function --function-name rag-text-to-sql --query Configuration.Architectures
```

If mismatched, rebuild and update:
```bash
docker build --platform linux/arm64 -f Dockerfile.lambda -t $ECR_URI:latest .
docker push $ECR_URI:latest
aws lambda update-function-code \
  --function-name rag-text-to-sql \
  --image-uri $ECR_URI:latest \
  --architectures arm64
```

**2. Verify API Gateway integration:**
```bash
# Get integration details
aws apigatewayv2 get-integrations --api-id $API_ID

# Verify the integration-uri matches your Lambda ARN
# Should be: arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:rag-text-to-sql
```

**3. Verify Mangum configuration in lambda_handler.py:**
```python
# Should include api_gateway_base_path="/prod"
handler = Mangum(app, lifespan="off", api_gateway_base_path="/prod")
```

**Reference**: See `DEPLOYMENT_FIXES.md` for complete troubleshooting steps.

---

### Problem: Cold start is too slow (10+ seconds)

**Symptom**: First request after idle period takes 10-15 seconds

**Solutions**:
1. **Provisioned Concurrency** (costs extra, but keeps Lambda warm):
   ```bash
   aws lambda put-provisioned-concurrency-config \
     --function-name rag-text-to-sql \
     --provisioned-concurrent-executions 1
   ```

2. **Scheduled Ping** (free, keeps Lambda warm):
   - Create EventBridge rule to invoke Lambda every 5 minutes
   - Cheaper than Provisioned Concurrency

3. **Optimize Docker image** (reduce startup time):
   - Already using multi-stage build ✓
   - Consider Lambda SnapStart (preview feature)

---

### Problem: Lambda runs out of /tmp space

**Error**: "No space left on device"

**Solution**:
1. Check current /tmp usage in logs
2. Clean up old files in Lambda handler:
   ```python
   import shutil

   # At start of each invocation
   if os.path.exists("/tmp/uploads"):
       shutil.rmtree("/tmp/uploads")
   os.makedirs("/tmp/uploads", exist_ok=True)
   ```
3. Increase ephemeral storage (up to 10GB max):
   ```bash
   aws lambda update-function-configuration \
     --function-name rag-text-to-sql \
     --ephemeral-storage Size=10240
   ```

---

### Problem: Health check shows all services as false

**Error**: Health endpoint returns `"embedding_service": false, "vector_service": false, ...`

**Symptom**: All services show as unavailable even though environment variables are configured correctly.

**Root Cause**: FastAPI startup events don't execute with Mangum when `lifespan="off"`. Services weren't initialized.

**Solution**: This has been fixed in the latest deployment. The `lambda_handler.py` now manually calls `initialize_services()` on container startup.

**Verify the fix:**
1. Check that `lambda_handler.py` contains:
   ```python
   from app.main import initialize_services
   initialize_services()
   ```

2. Check that `app/main.py` contains the `initialize_services()` function

3. Rebuild and redeploy:
   ```bash
   docker build --platform linux/arm64 -f Dockerfile.lambda -t $ECR_URI:latest .
   docker push $ECR_URI:latest
   aws lambda update-function-code \
     --function-name rag-text-to-sql \
     --image-uri $ECR_URI:latest \
     --architectures arm64
   ```

**Check CloudWatch logs** to verify services are initializing:
```bash
aws logs tail /aws/lambda/rag-text-to-sql --since 5m | grep "initialized"
```

You should see log messages like:
- "✓ Document RAG services initialized!"
- "✓ Text-to-SQL service initialized and trained!"
- "✓ Cache service initialized!"

**Reference**: See `DEPLOYMENT_FIXES.md` for complete details on this issue and solution.

---

### Problem: Database connection errors

**Error**: "could not connect to server", "too many connections"

**Solution**:
1. **Reuse connections** across Lambda invocations (already implemented ✓)
2. **Use connection pooling** with PgBouncer or RDS Proxy
3. **Increase database max connections** in Supabase settings
4. **Add retry logic** for transient connection failures

---

### Problem: API returns wrong results or errors

**Debug checklist:**

1. **Check environment variables**:
   ```bash
   aws lambda get-function-configuration \
     --function-name rag-text-to-sql \
     --query 'Environment.Variables'
   ```

2. **Check CloudWatch logs** for detailed errors:
   ```bash
   aws logs tail /aws/lambda/rag-text-to-sql --follow
   ```

3. **Test locally first**:
   ```bash
   docker build -f Dockerfile.lambda -t rag-lambda:test .
   docker run -p 9000:8080 \
     -e OPENAI_API_KEY="your-key" \
     -e PINECONE_API_KEY="your-key" \
     -e DATABASE_URL="your-url" \
     rag-lambda:test
   ```

4. **Check external services are reachable**:
   - OpenAI API key is valid
   - Pinecone index exists and is accessible
   - Supabase database is running

---

## FAQ

### Q: How much does this cost?

**A**: Approximately **$137-242/month** for 100K requests/month. See [Cost Breakdown](#cost-breakdown) for details.

---

### Q: Can I use a custom domain name?

**A**: Yes! Use AWS Route 53 + ACM:
1. Register domain in Route 53
2. Request SSL certificate in ACM
3. Create custom domain in API Gateway
4. Update Route 53 to point to API Gateway

**Cost**: ~$0.50/month (Route 53 hosted zone)

---

### Q: How do I add a staging environment?

**A**: Easiest approach:
1. Create a `staging` branch in Git
2. Create second Lambda function: `rag-text-to-sql-staging`
3. Create second API Gateway: `rag-api-staging`
4. Update `.github/workflows/deploy.yml` to deploy `staging` branch to staging Lambda

---

### Q: Can I deploy to a different AWS region?

**A**: Yes, just replace `us-east-1` with your region:
- `us-west-2` (Oregon)
- `eu-west-1` (Ireland)
- `ap-southeast-1` (Singapore)

**Note**: Some services may cost slightly more in non-US regions.

---

### Q: What happens to uploaded files?

**A**: Files stored in `/tmp` are deleted when Lambda container shuts down (after ~15 minutes of inactivity). This is by design for this deployment.

**To persist files**: Use S3 bucket (requires code changes and adds ~$0.023/GB/month cost).

---

### Q: How do I enable HTTPS only?

**A**: API Gateway HTTP APIs use HTTPS by default. HTTP requests are automatically redirected to HTTPS. No additional configuration needed! ✓

---

### Q: Can I see API usage analytics?

**A**: Yes! Two options:
1. **CloudWatch Metrics** (free):
   - Go to CloudWatch → Metrics → API Gateway
   - View request count, latency, errors

2. **X-Ray Tracing** (adds $5-10/month):
   ```bash
   aws lambda update-function-configuration \
     --function-name rag-text-to-sql \
     --tracing-config Mode=Active
   ```

---

### Q: How do I delete everything and start over?

**A**: Run these commands to delete all AWS resources:

```bash
# Delete Lambda function
aws lambda delete-function --function-name rag-text-to-sql

# Delete API Gateway
aws apigatewayv2 delete-api --api-id $API_ID

# Delete ECR repository (with all images)
aws ecr delete-repository \
  --repository-name rag-text-to-sql \
  --force

# Delete IAM role policies
aws iam detach-role-policy \
  --role-name rag-lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Delete IAM role
aws iam delete-role --role-name rag-lambda-execution-role
```

Then follow this guide again from Step 3.

---

## Summary

Congratulations! 🎉 You've successfully deployed your RAG + Text-to-SQL application to AWS Lambda.

**What you accomplished:**
- ✅ Created AWS Lambda function with ARM64 architecture, 8GB RAM, and 15-minute timeout
- ✅ Set up ECR container registry with cross-platform Docker builds
- ✅ Created API Gateway HTTP endpoint with /prod base path
- ✅ Configured service initialization for Mangum compatibility
- ✅ Fixed file permissions for Lambda runtime access
- ✅ Configured GitHub Actions for automatic deployments
- ✅ Enabled CloudWatch logging and monitoring

**Your application is now:**
- 🚀 **Serverless** - No servers to manage
- 💰 **Cost-effective** - Pay only for usage (~$127-227/month for 100K requests)
- ⚡ **ARM64 optimized** - 20% cheaper than x86_64, better performance
- ♾️ **Scalable** - Automatically scales to handle traffic
- 🔄 **Auto-deployed** - Push to `main` branch → Automatic deployment
- 🌍 **Cross-platform** - Builds work on Windows, Mac (Intel/ARM), Linux

**Key Technical Achievements:**
- ✓ ARM64 architecture for optimal cost/performance
- ✓ Manual service initialization bypassing Mangum lifespan limitations
- ✓ Proper file permissions (755 for dirs, 644 for .py files)
- ✓ API Gateway base path configuration for stage support
- ✓ Cross-platform Docker builds with --platform flag

**Next steps:**
- Monitor logs with CloudWatch
- Test all endpoints thoroughly
- Set up CloudWatch alarms for errors
- Consider adding a staging environment
- Optimize costs based on actual usage

**Reference Documentation:**
- `DEPLOYMENT_FIXES.md` - Complete troubleshooting guide
- `CROSS_PLATFORM_BUILD.md` - Building on different platforms
- `TEAM_SETUP.md` - Setup guide for team members

**Need help?** Check the [Troubleshooting](#troubleshooting) section or review CloudWatch logs.

---

## Additional Resources

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [API Gateway HTTP API Guide](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [AWS CLI Command Reference](https://awscli.amazonaws.com/v2/documentation/api/latest/index.html)

---

**Last Updated**: 2026-01-16
**Deployment Type**: AWS Lambda with Container Image (ARM64)
**Architecture**: ARM64 (20% cheaper, better performance)
**CI/CD**: GitHub Actions
**Deployment Time**: ~10-15 minutes per deploy
**Key Features**: Service initialization fix, cross-platform builds, file permissions
