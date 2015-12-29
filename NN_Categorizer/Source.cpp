#include <iostream>
#include "FANN-2.2.0-Source\FANN-2.2.0-Source\src\include\doublefann.h"

using namespace std;

int main() {
	cout << "Running" << endl;

	double inputs[3];
	double outputs[3];

	struct fann *ann = fann_create_standard(3, 3, 3, 3);

	int maxTrainingEpochs = 1000000;
	float errorLimit = 0.0001f;

	cout << "Mean Square Error: " << fann_get_MSE(ann) << endl;

	int epoch = 0;

	do
	{
		inputs[0] = 1;
		inputs[1] = 1;
		inputs[2] = 0;

		outputs[0] = 0;
		outputs[1] = 1;
		outputs[2] = 1;

		fann_train(ann, inputs, outputs);

		inputs[0] = 0;
		inputs[1] = 1;
		inputs[2] = 1;

		outputs[0] = 1;
		outputs[1] = 0;
		outputs[2] = 1;

		fann_train(ann, inputs, outputs);

		inputs[0] = 1;
		inputs[1] = 0;
		inputs[2] = 1;

		outputs[0] = 1;
		outputs[1] = 1;
		outputs[2] = 0;

		fann_train(ann, inputs, outputs);

		if (epoch % 100 == 0)
		{
			cout << "Epoch: " << epoch << endl;
			cout << "Mean Square Error: " << fann_get_MSE(ann) << endl;
		}

		epoch++;
	} while (epoch < maxTrainingEpochs && fann_get_MSE(ann) > errorLimit);

	if (fann_save(ann, "neuralnet_bitshift.data"))
		cout << "Saved succesfully!" << endl;
	else
		cout << "Could not save file" << endl;
}