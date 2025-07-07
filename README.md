
# PyTorch ANN Classifier

This project is a beginner-friendly implementation of an **Artificial Neural Network (ANN)** using **PyTorch**. It demonstrates how to:

* Prepare and preprocess data
* Build a simple neural network
* Train the network
* Evaluate its performance

The project uses synthetic data for binary classification and is structured as a Jupyter Notebook for ease of understanding and experimentation.

---

##  File Structure

* `AnnUsingPytorch.ipynb` – Main notebook containing all the code for data generation, model creation, training, and evaluation.

---

##  What the Project Does

The notebook walks through:

1. **Generating synthetic classification data** using `sklearn.datasets.make_classification`.
2. **Splitting the data** into training and testing sets.
3. **Converting data into PyTorch tensors** and loading it with `DataLoader`.
4. **Defining a neural network model** using `torch.nn`.
5. **Training the model** using `BCELoss` and the Adam optimizer.
6. **Evaluating the model** by checking its accuracy on the test set.
7. **Visualizing the loss** over epochs using matplotlib.

---

##  Tech Stack

* Python
* PyTorch
* Scikit-learn
* Matplotlib
* NumPy

---

##  How to Run

### Prerequisites

Make sure the following Python libraries are installed:

```bash
pip install torch torchvision scikit-learn matplotlib
```

### Run the Notebook

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Open the notebook:

   ```bash
   jupyter notebook AnnUsingPytorch.ipynb
   ```

3. Run the cells step-by-step to train and evaluate the neural network.

---

##  Key Concepts Covered

* Basic structure of a neural network in PyTorch
* Activation functions (ReLU, Sigmoid)
* Binary classification with synthetic data
* Training loop logic: forward pass, loss computation, backward pass, optimizer step
* Using `DataLoader` for batching
* Model evaluation with accuracy metric

---

## Output Example

The model is trained for several epochs and prints the loss at regular intervals. A plot is also generated to show the loss over time. The final output includes the model's accuracy on test data.

---

## Use Case

This notebook is ideal for:

* Beginners looking to learn PyTorch
* Students or professionals brushing up on neural network fundamentals
* Anyone looking for a simple template to start their own classification model

---

##  Feedback

Feel free to open an issue if you find a bug or want to suggest improvements.



