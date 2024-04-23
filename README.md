# EEG-Based Emotion Recognition Using Deep Learning

This repository contains a Python-based project for emotion recognition using EEG (Electroencephalogram) data. The project utilizes deep learning techniques to analyze EEG brainwave data for identifying emotional states.

## Project Overview

The code in this repository is developed to handle EEG data processing, model training, and evaluation for emotion recognition. The project is structured into several modules:

- `data_processing.py`: For loading and preprocessing EEG dataset.
- `model.py`: Contains the architecture of the deep learning model.
- `training.py`: Manages the training process of the model.
- `evaluation.py`: For evaluating the model's performance and plotting results.
- `main.py`: The entry point to run the entire pipeline.

Developed in **Python 3.11**, the project is designed to be modular and scalable.

## Setup and Installation

To set up the project:

1. **Clone the Repository**

2. **Install Dependencies**
   All dependencies are listed in `requirements.txt`. Install them using:

```bash
  pip install -r requirements.txt
```

3. **Run the Project**
   Execute the main script to start the process:

```bash
python main.py
```

## Dataset Summary

- **Subjects**: The dataset was collected from two individuals, one male and one female.
- **Measurement Duration**: Data for three emotional states—positive, neutral, and negative—was collected for 3 minutes each per state. Additionally, 6 minutes of resting data in a neutral state were recorded.
- **Equipment Used**: The Muse EEG headband was employed to record data from the TP9, AF7, AF8, and TP10 electrode placements using dry electrodes.

## Emotional Stimuli

Data collection involved exposing subjects to various media clips to elicit positive and negative emotional states:

**Negative Emotions:**
- *Marley and Me* (Twentieth Century Fox) - Death Scene
- *Up* (Walt Disney Pictures) - Opening Death Scene
- *My Girl* (Imagine Entertainment) - Funeral Scene

**Positive Emotions:**
- *La La Land* (Summit Entertainment) - Opening musical number
- *Slow Life* (BioQuest Studios) - Nature timelapse
- *Funny Dogs* (MashupZone) - Compilation of funny dog videos

## Results

![image](https://github.com/fraware/Emotions_Predictions/assets/113530345/6380d2a7-c153-4d18-9e14-1743dc1a9a83)

## References

This project references several studies and papers for EEG data analysis and deep learning approaches:

1. Cahn, B. R., & Polich, J. (2006). Meditation states and traits: EEG, ERP, and neuroimaging studies. Psychological Bulletin, 132(2), 180-211. [Link](https://doi.org/10.1037/0033-2909.132.2.180)
2. Baijal, S., & Srinivasan, N. (2010). Theta activity and meditative states: spectral changes during concentrative meditation. Cognitive Processing, 11(1), 31-38. [Link](https://doi.org/10.1007/s10339-009-0272-0)
3. DeLosAngeles, D., et al. (2016). Electroencephalographic correlates of states of concentrative meditation. International Journal of Psychophysiology, 110, 27-39. [Link](https://doi.org/10.1016/j.ijpsycho.2016.09.020)
4. Deolindo, C. S., et al. (2020). A critical analysis on characterizing the meditation experience through the electroencephalogram. Frontiers in Systems Neuroscience, 14, 53. [Link](https://doi.org/10.3389/fnsys.2020.00053)
5. Händel, B. F., et al. (2011). Alpha oscillations correlate with the successful inhibition of unattended stimuli. Journal of Cognitive Neuroscience, 23(9), 2494-2502. [Link](https://doi.org/10.1162/jocn.2010.21557)
6. Kaur, C., & Singh, P. (2015). EEG derived neuronal dynamics during meditation: Progress and challenges. Advances in Preventive Medicine, 2015, 614723. [Link](https://doi.org/10.1155/2015/614723)
