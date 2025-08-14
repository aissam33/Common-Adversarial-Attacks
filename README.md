# Common-Adversarial-Attacks
This project tests the vulnerability of a neural network model to adversarial attacks using the Fast Gradient Sign Method (FGSM). The method introduces small, human-imperceptible perturbations to input data, which can lead the model to make incorrect predictions
<img width="756" height="251" alt="image" src="https://github.com/user-attachments/assets/0390e994-6c6f-4d44-aad7-8ece71c6873b" />


The project has been integrated into Flask to make it easier to use FGSM and perform tests on the pre-trained model. Simply start app.py, upload the first image (which produces correct results), and then upload the second, adversarially perturbed image. Although it visually appears identical to the first, it causes the model to produce incorrect predictions. 
# Generate your own adversarial images
To generate your own adversarial images, simply use the provided image adversariale fgsm.py code for the pre-trained model, available at the following link [Here](https://drive.google.com/file/d/18zWzZjQnA9_tjr49CANJZ2LOQv6VFoDr/view?usp=sharing)
