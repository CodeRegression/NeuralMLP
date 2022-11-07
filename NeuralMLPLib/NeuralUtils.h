//--------------------------------------------------
// A set of utilities for evaluating the performance of classification programs
//
// @author: Wild Boar
//
// @date: 2022-09-26
//--------------------------------------------------

#pragma once

#include <fstream>
#include <iostream>
using namespace std;

#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include <NVLib/StringUtils.h>

#include "TrainData.h"

namespace NVL_AI
{
	class NeuralUtils
	{
	public:
		static void WriteData(const string& path, const string& name, const string& description, Mat& data);
		static TrainData * LoadData(const string& path);
		static Ptr<ml::ANN_MLP> CreateNetwork(const string structure, int inputCount, int outputCount = 1);
	private:
		static Mat LoadARFF(const string& path, vector<string>& fieldNames);
		static void RenderHeader(ostream& writer, const string& name, const string& description, int paramCount);
		static void RenderData(ostream& writer, Mat& data); 
	};
}
