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
    _network = NVL_AI::NeuralUtils::CreateNetwork(networkConfig, _trainData->GetInputs().cols, _trainData->GetOutputs().cols);
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
    _logger->Log(1, "Execution not yet implemented");

}