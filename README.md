# Deep Convolutional Q-Learning for Pac-Man

This repository contains a complete implementation of a Deep Convolutional Q-Learning (DCQN) agent designed to play the Pac-Man game. Using deep learning and reinforcement learning techniques, the agent learns optimal strategies for navigating the game environment, avoiding ghosts, and maximizing the score.

<a target="_blank" href="https://colab.research.google.com/github/vinaysanga/Pac-Man-using-Reinforcement-Learning/blob/master/PacMan.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Getting Started

### Prerequisites

- To run this project, you'll need the following installed:

    - Python 3.8 or newer
    - PyTorch
    - Gymnasium
    - Pillow (PIL)
    - torchvision
    - imageio with ffmpeg support

- You can install all the necessary packages using the following commands, if you want to run the .py script:

    ```bash
    pip install gymnasium "gymnasium[atari, accept-rom-license]" gymnasium[box2d] torch gym pillow torchvision imageio[ffmpeg]
    ```

- If you are using the jupyter notebook, running the first cell will install the dependencies for you.

## Running the Project
To start working with the project, open the `Deep_Convolutional_Q_Learning_for_Pac_Man_Complete_Code.ipynb` notebook in Jupyter:

```bash
jupyter notebook Deep_Convolutional_Q_Learning_for_Pac_Man_Complete_Code.ipynb
```

Follow the instructions in the notebook to train your agent.

## Model Architecture

The model uses a Convolutional Neural Network (CNN) that takes in the preprocessed frames from the Pac-Man game as input and outputs a set of action values for the agent to take. The network architecture consists of several convolutional layers followed by fully connected layers.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
