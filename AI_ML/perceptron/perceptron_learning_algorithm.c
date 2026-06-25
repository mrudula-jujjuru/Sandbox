// perceptron_learning_algorithm.c
// student performance prediction based on study hours and attendance using a perceptron learning algorithm
#include <stdio.h>
#define EPOCHS 20000
#define LR 0.1
#define DATA_SIZE 6
// Dataset: [study_hours, attendance], labels: pass (1) or fail (0)
float X[DATA_SIZE][2] = {
 {2, 50},
 {5, 60},
 {8, 85},
 {6, 80},
 {1, 30},
 {9, 90}
};
int y[DATA_SIZE] = {0, 0, 1, 1, 0, 1};
float weights[2] = {0, 0};
float bias = 0;
int step_function(float x) {
 return x >= 0 ? 1 : 0;
}
void train() {
 for (int epoch = 0; epoch < EPOCHS; epoch++) {
 printf("\nEpoch %d\n", epoch + 1);
 for (int i = 0; i < DATA_SIZE; i++) {
 float x1 = X[i][0];
 float x2 = X[i][1];
 int target = y[i];
 float linear_output = x1 * weights[0] + x2 * weights[1] + bias;
 int prediction = step_function(linear_output);
 int error = target - prediction;
 weights[0] += LR * error * x1;
 weights[1] += LR * error * x2;
 bias += LR * error;
 }
 printf("Weights: [%f, %f], Bias: %f\n", weights[0], weights[1], bias);
 }
}
void test_training_data() {
 printf("\nFinal Predictions on Training Data:\n");
 for (int i = 0; i < DATA_SIZE; i++) {
 float x1 = X[i][0];
 float x2 = X[i][1];
 float result = x1 * weights[0] + x2 * weights[1] + bias;
 int prediction = step_function(result);
 printf("Input: [%.1f, %.1f], Predicted: %d, Actual: %d\n", x1, x2, prediction, y[i]);
 }
}
void test_user_input() {
 float study_hours, attendance;
 while (1) {
 printf("\nEnter study hours (-1 to exit): ");
 scanf("%f", &study_hours);
 if (study_hours == -1) break;
 printf("Enter attendance (%%): ");
 scanf("%f", &attendance);
 float result = study_hours * weights[0] + attendance * weights[1] + bias;
 int prediction = step_function(result);
 printf("Prediction: %s\n", prediction == 1 ? "PASS" : "FAIL");
 }
}
int main() {
 train();
 test_training_data();
 test_user_input();
 return 0;
}