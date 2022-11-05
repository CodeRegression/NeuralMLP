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

#include <opencv2/opencv.hpp>
using namespace cv;

#include <NVLib/StringUtils.h>

namespace NVL_AI
{
	class ScoreUtils
	{
	public:
		static Mat LoadARFF(const string& path, vector<string>& fieldNames);
	};
}
