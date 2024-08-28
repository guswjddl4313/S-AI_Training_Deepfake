# Subject - Deepfake Detector with Automatic learning
- Motivation & Purpose
    - The necessity of Deepfake Detector due to recently resurfaced deepfake-related crimes
# Architecture
- Front Face Photo Dataset Learning
- Deepfake internally after collecting various front-facing photos from web browsers
    - New Datasets Can Be Built

1. Construct a list of deepfakes randomly from the collected front photos (50:50)
2. Make a deepfake with a pair taken from each list
3. Add processed dataset (100)

- **Using ControlNet** (Use Stable-Diffusion instead of Flux due to cost issues)
    - Change parts other than the face → It doesn't have to be ControlNet

---
# If possible
- Learn face photography according to angle(?)
- Facial expression learning X
- Learning to determine if the frame connection is natural if you want to include video as well → Take a long time