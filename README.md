# Agentic-Chatbot-APP-CICD-Deployment-with-Github-Actions-on-AWS

## Description: About the deployment
1. Build docker image of the source code
2. Push your docker image to docker hub
3. Launch Your EC2 
4. Pull Your image from docker hub in EC2
5. Launch your docker image in EC2

---

## 1. Login to AWS console.
Go to [https://aws.amazon.com/console/](https://aws.amazon.com/console/) and sign in with your AWS root account or admin IAM credentials.

---

## 2. Create IAM user for deployment
1. Go to **IAM** service in AWS Console.
2. Click **Users** ➡️ **Create user**.
3. Enter User name (e.g. `agentic-chatbot-deployer`).
4. Select **Attach policies directly**.

### #Policy:
- **`AmazonEC2FullAccess`**

5. Click **Create user**.
6. Select the newly created user ➡️ **Security credentials** tab.
7. Under **Access keys**, click **Create access key** ➡️ Choose **Command Line Interface (CLI)**.
8. Download or copy your:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

---

## 3. Create EC2 machine (Ubuntu)
1. Go to **EC2** service ➡️ **Instances** ➡️ **Launch instances**.
2. **Name**: `agentic-chatbot-server`.
3. **Application and OS Images**: **Ubuntu** (Ubuntu Server 22.04 LTS or 24.04 LTS, 64-bit x86).
4. **Instance type**: `t2.medium` or `t3.medium` (recommended: at least 2 vCPUs, 4GB RAM).
5. **Key pair**: Select or create a new key pair (`.pem` file) to SSH into the instance.
6. **Network settings**:
   - Check **Allow SSH traffic from Anywhere**.
   - Under **Security Group rules**, click **Add security group rule**:
     - **Type**: Custom TCP
     - **Port range**: `8501`
     - **Source type**: Anywhere (`0.0.0.0/0`)
     > **Note: Do the port mapping to this port:- 8501**
7. **Storage**: At least 20–30 GB General Purpose SSD (gp3).
8. Click **Launch instance**.

---

## 4. Open EC2 and Install docker in EC2 Machine:
Connect to your EC2 instance via SSH:
```bash
ssh -i "your-key.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>
```

Run these commands inside your EC2 terminal:

```bash
# optional
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo usermod -aG docker ubuntu
newgrp docker
```

> **Note: Do the port mapping to this port:- 8501**
> Verify Docker installation:
> ```bash
> docker --version
> ```

---

## 5. Configure EC2 as self-hosted runner:
In your GitHub repository:
**Settings** > **Actions** > **Runners** > **New self-hosted runner** > Choose **Linux** (OS) > then run command one by one in your EC2 terminal:

```bash
# 1. Create a folder for the actions runner
mkdir actions-runner && cd actions-runner

# 2. Download the latest runner package (use the exact command from your GitHub page)
curl -o actions-runner-linux-x64-2.322.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.322.0/actions-runner-linux-x64-2.322.0.tar.gz

# 3. Extract the installer
tar xzf ./actions-runner-linux-x64-2.322.0.tar.gz

# 4. Configure the runner (paste the exact config command provided in your GitHub page with your token)
./config.sh --url https://github.com/<your-username>/<your-repo-name> --token <YOUR_RUNNER_TOKEN>

# 5. Install and start as a 24/7 background system service
sudo ./svc.sh install
sudo ./svc.sh start
```

Verify runner service status:
```bash
sudo ./svc.sh status
```
*(On your GitHub repository page under **Settings > Actions > Runners**, you will see your runner in green **Idle** status).*

---

## 6. How to add secret keys to GitHub Actions:
setting>actions>runner>new self hosted runner> choose os> then run command one by one

In your GitHub repository:
**Settings** > **Secrets and variables** > **Actions** > **New repository secret**

Add the following secret keys:

```text
REGISTRY=docker.io
DOCKER_USERNAME=your-dockerhub-username
DOCKER_PASSWORD=your-dockerhub-access-token
IMAGE_NAME=agentic-chatbot
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
GEMINI_API_KEY=your-gemini-api-key
TAVILY_API_KEY=your-tavily-api-key
WEATHER_STACK_API_KEY=your-weatherstack-api-key
ALPHA_VANTAGE_API_KEY=your-alphavantage-api-key
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=Agentic-ChatBot
```

> **Note on Keys**:
> - `GEMINI_API_KEY` (or `GOOGLE_API_KEY`): Used by Google Generative AI for Chat & RAG Embeddings (No OpenAI key needed).
> - `WEATHER_STACK_API_KEY`: Used by WeatherStack API for live weather lookup.
> - `ALPHA_VANTAGE_API_KEY`: Used for real-time stock price lookup.
> - `TAVILY_API_KEY`: Used for real-time web search.

---

## 7. Automated CI/CD Pipeline Flow

Whenever you push code changes to the `main` branch (`git push origin main`):

1. **GitHub Runner (`ubuntu-latest`)**:
   - Builds the Docker image from source.
   - Logs in to Docker Hub.
   - Pushes the image: `<DOCKER_USERNAME>/agentic-chatbot:latest`.

2. **AWS EC2 Runner (`self-hosted`)**:
   - Pulls the newly built image from Docker Hub.
   - Stops and removes any existing container.
   - Launches the new container on port `8501` with persistent volume:
     ```bash
     docker run -d \
       --name agentic-chatbot-app \
       --restart unless-stopped \
       -p 8501:8501 \
       -v agentic_chatbot_data:/app/persistent_data \
       -e DATA_DIR=/app/persistent_data \
       ... (all secrets injected)
     ```
   - Cleans up dangling images (`docker image prune -f`).

---

## 8. Access the Running Application
Open your browser and navigate to:
```text
http://<YOUR_EC2_PUBLIC_IP>:8501
```
