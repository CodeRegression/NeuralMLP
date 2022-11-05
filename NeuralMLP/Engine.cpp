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
}

/**
 * Main Terminator 
 */
Engine::~Engine() 
{
    delete _parameters;
}

//--------------------------------------------------
// Execution Entry Point
//--------------------------------------------------

/**
 * Entry point function
 */
void Engine::Run()
{
    _logger->Log(1, "Loading input file");
    auto dataPath = ArgUtils::GetString(_parameters, "input"); auto fields = vector<string>();
    Mat data = NVL_AI::ScoreUtils::LoadARFF(dataPath, fields);
    if (data.empty()) throw runtime_error("DAta load failed");
    _logger->Log(1, "Extracted %i records with %i fields", data.rows, fields.size());

}