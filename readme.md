# DDPG Pendulum-v1

This project implements the **Deep Deterministic Policy Gradient (DDPG)** reinforcement learning algorithm using PyTorch to solve the Gymnasium **Pendulum-v1** environment.

The agent is trained to swing up and stabilize the inverted pendulum.

## Project Structure

* `DDPG_Pendulum.py` / `main.py`: Primary script(s) containing the DDPG agent definition, network architectures, and the training loop.
* `ddpg_demo.py`: Script used for loading the pre-trained weights and visualizing the agent's performance.
* `requirements.txt`: Lists all Python package dependencies (PyTorch, Gymnasium, NumPy, etc.).
* `models/`: Directory containing the saved, **pre-trained weights (`weights.pth`)**.
* `.gitignore`: Configured to ignore environment files and compiled code, but **includes** the model weights.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <YOUR_REPO_URL>
    cd DDPG-Pendulum-v1
    ```

2.  **Create and Activate Virtual Environment:**
    The project relies on a virtual environment named `.venv`.
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Linux/macOS
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Agent

### 1. Run the Demo / Evaluation

To visualize the pre-trained agent's performance immediately, execute the demo script:

```bash
python ddpg_demo.py