//--------------------------------------------------
// Implementation code for the Engine
//
// @author: Wild Boar
//
// @date: 2022-11-05
//--------------------------------------------------

#include "Engine.h"
using namespace NVL_App;

//--------------------------------------------------
// Constructor and Terminator
//--------------------------------------------------

/**
 * Main Constructor
 * @param logger The logger that we are using for the system
 * @param parameters The input parameters
 */
Engine::Engine(NVLib::Logger* logger, NVLib::Parameters* parameters) 
{
    _logger = logger; _parameters = parameters;

    _logger->Log(1, "Loading training data");
    auto dataPath = ArgUtils::GetString(parameters, "input");
    _trainData = NVL_AI::NeuralUtils::LoadData(dataPath);

    _logger->Log(1, "Setup the given network");
    auto networkConfig = ArgUtils::GetString(parameters, "ann_config");
    _learnRate = ArgUtils::GetDouble(parameters, "learn_rate");
    _network = NVL_AI::NeuralUtils::CreateNetwork(networkConfig, _learnRate, _trainData->GetInputs().cols, _trainData->GetOutputs().cols);
    _iterations = ArgUtils::GetInteger(parameters, "iterations");
    _outputPath = ArgUtils::GetString(parameters, "output");

}

/**
 * Main Terminator 
 */
Engine::~Engine() 
{
    delete _parameters; 
    if (_trainData != nullptr) delete _trainData;
}

//--------------------------------------------------
// Execution Entry Point
//--------------------------------------------------

/**
 * Entry point function
 */
void Engine::Run()
{
    _logger->Log(1, "Initialize Training");
    auto train = ml::TrainData::create(_trainData->GetInputs(), ml::ROW_SAMPLE, _trainData->GetOutputs());
	_network->train(train);

	_logger->Log(1, "Starting training");
	auto bestScore = NVL_AI::NeuralUtils::GetScore(_trainData, _network);
    _logger->Log(1, "Initial Score: %f", bestScore);
    for (auto i = 0; i < _iterations; i++) 
	{
		_network->train(train,  ml::ANN_MLP::UPDATE_WEIGHTS);
		auto current = NVL_AI::NeuralUtils::GetScore(_trainData, _network);
		_logger->Log(1, "Iteration %i: %f", i, current);

        if (current < bestScore) 
        {
            _logger->Log(1, "Best result so far, saving");
            NVL_AI::NeuralUtils::Save(_outputPath, _network);
            bestScore = current;
            if (bestScore < 1e-4) 
            {
                _logger->Log(1, "Low score found, terminating!");
            }
        }
	}
}