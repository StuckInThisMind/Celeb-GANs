# Generative Adversarial Networks (GANs) - Experiment 3

## Overview
This Jupyter Notebook contains the implementation and analysis of Generative Adversarial Networks (GANs) as part of Experiment 3. The notebook explores the training process, model architecture, and generated outputs.

## Requirements
To run this notebook, ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- TensorFlow / PyTorch (depending on the framework used)
- NumPy
- Matplotlib
- Other dependencies as specified in the notebook

Use the following command to install missing dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preprocessing
1. **Load Dataset**: The dataset is loaded from a specified source (e.g., MNIST, CIFAR-10, or custom dataset).
2. **Normalization**: Data is normalized to ensure stable training (e.g., scaling pixel values to [0,1] or [-1,1]).
3. **Batching**: The dataset is divided into mini-batches for efficient training.
4. **Augmentation (Optional)**: Data augmentation techniques may be applied to improve model robustness.

## Model Training
To train the GAN model, follow these steps:
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook gans-exp-3(1).ipynb
   ```
2. Run the notebook cells sequentially.
3. Adjust hyperparameters such as learning rate, batch size, and number of epochs in the training section.
4. The training loop will alternate updates between the Generator and Discriminator models.
5. Monitor loss values and generated images to evaluate training progress.

## Model Testing
- After training, the Generator model can generate new samples.
- Run the testing section of the notebook to visualize the generated outputs.
- Optionally, evaluate the quality of generated images using metrics such as Inception Score (IS) or Frechet Inception Distance (FID).

## Expected Outputs
- The Generator should produce realistic images resembling the training dataset.
- During early epochs, generated images may appear noisy or unstructured.
- As training progresses, image quality should improve, showing more defined features and structures.
- Final outputs should closely match the original dataset distribution.

## Notes
- Training GANs requires significant computational power; running on a GPU is recommended.
- Adjust the learning rate, batch size, and number of epochs for improved results.

## Author
Aaradhya Badal

## License
This project is licensed under the MIT License.

