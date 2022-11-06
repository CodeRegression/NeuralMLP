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

    // Generate the network layout
    Mat_<int> layers(7,1);
    layers(0) = data.cols - 1; // input
    layers(1) = (data.cols - 1); // hidden
    layers(2) = (data.cols - 1); // hidden
    layers(3) = (data.cols - 1); // hidden
    layers(4) = (data.cols - 1); // hidden
    layers(5) = (data.cols - 1); // hidden
    layers(6) = 1; // output, 1 pin per class.

    // Setup the network
    auto ann = ml::ANN_MLP::create();
    ann->setLayerSizes(layers);
    ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0, 0);
    ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.0001));
    ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);

    _logger->Log(1, "Building the training data");
    auto input = (double *) data.data;
    Mat train_in = Mat_<float>(data.rows, data.cols - 1);
    Mat train_out = Mat_<float>(data.rows, 1);

    for (auto row = 0; row < data.rows; row++) 
    {
        for (auto column = 0; column < data.cols; column++) 
        {
            auto index = column + row * data.cols;
            auto value = input[index];
         
            if (column < data.cols - 1)
            {        
                auto i = column + row * train_in.cols;
                ((float *)train_in.data)[i] = value;
            }
            else 
            {
                ((float *)train_out.data)[row] = (float)value;
            }
        }
    }

    auto tdata = ml::TrainData::create(train_in, ml::ROW_SAMPLE, train_out);

    cout << "Loop Test" << endl;
    ann->train(tdata);

    for (auto j = 0; j < 100; j++) 
    {
        cout << "Iteration: " << j << endl;
        ann->train(tdata, ml::ANN_MLP::UPDATE_WEIGHTS);

        // Create container matrices
        Mat input = Mat_<float>(1, train_in.cols);
        Mat output = Mat_<float>(1, 1);

        // Check the result
        auto total = 0.0;
        for (auto row = 0; row < data.rows; row++) 
        {
            for (auto column = 0; column < train_in.cols; column++) 
            {
                auto index = column + row * train_in.cols;
                ((float *)input.data)[column] = ((float *) train_in.data)[index];
            }

            auto expected = ((float *)train_out.data)[row];

            ann->predict(input, output);

            auto predicted = ((float *)output.data)[0];
            auto difference = predicted - expected;
            total += abs(difference);    
        }

        cout << "Error: " << total << " aka " << (total / data.rows) << endl;
    }

    auto writer = FileStorage("model.xml", FileStorage::WRITE | FileStorage::FORMAT_XML);
    ann->write(writer);
    writer.release();
}