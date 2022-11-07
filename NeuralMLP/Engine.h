//--------------------------------------------------
// Defines a basic engine for a vanilla C++ project.
//
// @author: Wild Boar
//
// @date: 2022-11-05
//--------------------------------------------------

#pragma once

#include <iostream>
using namespace std;

#include <NVLib/Logger.h>

#include <NeuralMLPLib/ArgUtils.h>
#include <NeuralMLPLib/NeuralUtils.h>

namespace NVL_App
{
	class Engine
	{
	private:
		NVLib::Parameters * _parameters;
		NVLib::Logger* _logger;

		NVL_AI::TrainData * _trainData;
		Ptr<ml::ANN_MLP> _network;
		int _iterations;
		string _outputPath;
		double _learnRate;
	public:
		Engine(NVLib::Logger* logger, NVLib::Parameters * parameters);
		~Engine();

		void Run();
	};
}
