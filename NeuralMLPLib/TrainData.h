//--------------------------------------------------
// The training data for training the neural network
//
// @author: Wild Boar
//
// @date: 2022-11-06
//--------------------------------------------------

#pragma once

#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

namespace NVL_AI
{
	class TrainData
	{
	private:
		Mat _inputs;
		Mat _outputs;

	public:
		TrainData(Mat& inputs, Mat& outputs) :
			_inputs(inputs), _outputs(outputs) {}

		inline Mat& GetInputs() { return _inputs; }
		inline Mat& GetOutputs() { return _outputs; }
	};
}
