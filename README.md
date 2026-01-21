# Common-Adversarial-Attacks
This project tests the vulnerability of a neural network model to adversarial attacks using the Fast Gradient Sign Method (FGSM). The method introduces small, human-imperceptible perturbations to input data, which can lead the model to make incorrect predictions
<img width="756" height="251" alt="image" src="https://github.com/user-attachments/assets/0390e994-6c6f-4d44-aad7-8ece71c6873b" />


The project has been integrated into Flask to make it easier to use FGSM and perform tests on the pre-trained model. Simply start app.py, upload the first image (which produces correct results), and then upload the second, adversarially perturbed image. Although it visually appears identical to the first, it causes the model to produce incorrect predictions. 
# Generate your own adversarial images
To generate your own adversarial images, simply use the provided image adversariale fgsm.py code for the pre-trained model, available at the following link [Here](https://drive.google.com/file/d/18zWzZjQnA9_tjr49CANJZ2LOQv6VFoDr/view?usp=sharing)
#DeepFool-Adversarial-Attack
This project evaluates the robustness of a deep neural network against DeepFool adversarial attacks.
DeepFool is an iterative, gradient-based method designed to find the minimal perturbation required to change a model’s prediction. The perturbation is typically imperceptible to the human eye, yet sufficient to fool a high‑performance classifier.
Unlike single-step attacks such as FGSM, DeepFool progressively approximates the decision boundary of the network by locally linearizing the classifier and computing the smallest displacement that crosses this boundary.
<img width="756" height="251" alt="image" src="https://github.com/user-attachments/assets/0390e994-6c6f-4d44-aad7-8ece71c6873b" />
#How it works
The attack is applied to a pre-trained ResNet‑34 model from torchvision.
Given an input image, DeepFool:
Computes the model’s original prediction
Iteratively estimates the closest decision boundary using class-wise gradients
Applies the smallest perturbation required to cross that boundary
Stops once the predicted label changes or a maximum number of iterations is reached
The resulting adversarial image remains visually indistinguishable from the original but leads the network to a different and incorrect classification.
#Visual analysis
The project provides:
Side-by-side visualization of the original and adversarial images
The final predicted labels before and after the attack
The number of iterations required to fool the model
A heatmap of the perturbation (amplified for visibility)
The L2 norm of the perturbation, highlighting its minimal magnitude
#Generate your own adversarial images
1.To generate your own DeepFool adversarial examples:
2.Load a pre-trained ResNet‑34 model
3.Provide an input image
4.Run the deepfool function included in the script
5.Visualize and analyze the perturbed output
The provided implementation supports configurable parameters such as:
Number of target classes,Overshoot factor,Maximum number of iterations
This makes it easy to experiment with different attack strengths and observe their impact on model robustness.
