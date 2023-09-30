# SyntheticEye
Advanced AI-Face detector
![SyntheticEye Logo](assets/Logo.png)

SyntheticeEye is developing free and reliable detectors for AI-generated content.

## Models
### Aletheia
![Aletheia Visualization](assets/Aletheia.png)
Aletheia is a machine learning model designed to detect AI-generated faces and distinguish them from real faces. This model can be used on our [website](syntheticeye.dev). Aletheia has achieved high accuracy in detecting images from various face generators, but we are still working on improving this model further and making it more robust.

### Argus (Beta)
We are currently developing Argus, a detector for AI-generated images in general. This model aims to detect a broad spectrum of AI-generated images. The Beta-Version of Argus can be tested on our [webiste](syntheticeye.dev)

### Future Models
We are continuously working on developing new models that are able to detect more types of content. These are a few detectors we plan to develop in the future:
- Text Detection
- Audio Detection
- Video Detection

## Potential
With our models, we want to provide everybody with reliable, free, and user-friendly tools that help them detect AI-Generated Content. Our models help to verify content and detect misinformation. With the rapid advancements of generative AI, tools like ours become essential to preserve trust.

## Repository Structure
- **assets/**: Images for the project.
- **notebooks/**: Jupyter notebooks for our models.
  - **runs/**: Contains logs from the notebooks run. These are not included in the repo, to ensure a lightweight repository.
- **scripts/**: Contains primarily helper functions.
- **state_dicts/**: Here we store our models state_dicts temporarily. They are not included in the repo, due to their large file size.
- **documentation**: Contains requirements for our models.
  - **Aletheia**: Requirements for Aletheia.
  - **Argus** Requirements for Argus.