//--------------------------------------------------
// Implementation of class ScoreUtils
//
// @author: Wild Boar
//
// @date: 2022-09-26
//--------------------------------------------------

#include "NeuralUtils.h"
using namespace NVL_AI;

//--------------------------------------------------
// Write ARFF File
//--------------------------------------------------

/**
 * @brief Write the given data to disk
 * @param path The path that we are writing to
 * @param name The name of the relation that we are processing
 * @param description A description of the relation that we are processing
 * @param data The data that we are writing
 */
void NeuralUtils::WriteData(const string& path, const string& name, const string& description, Mat& data) 
{
	// Create a writer
	auto writer = ofstream(path);

	// Render a header
	RenderHeader(writer, name, description, data.cols - 1);

	// Render body
	RenderData(writer, data);

	// Close the file
	writer.close();
}

/**
 * @brief The header of the ARFF file 
 * @param writer The writer that we are dealing with
 * @param name The name of the relation that we are saving
 * @param description The description of the project
 * @param paramCount The number of inputs parameters
 */
void NeuralUtils::RenderHeader(ostream& writer, const string &name, const string& description, int paramCount) 
{
    writer << "%----------------------------------------------" << endl;
    writer << "% " << description << endl;
    writer << "%" << endl;
    writer << "% @author: NeuralMLP " << endl;
    writer << "%----------------------------------------------" << endl;
    writer << endl;
    writer << "@RELATION " << name << endl;
    writer << endl;

    for (auto i = 0; i < paramCount; i++) writer << "@ATTRIBUTE p[" << i << "] REAL" << endl;
    writer << "@ATTRIBUTE class REAL" << endl << endl;
}

/**
 * @brief Write the outputs to disk
 * @param writer The writer that we are using 
 * @param data The data that we are writing
 */
void NeuralUtils::RenderData(ostream& writer, Mat& data) 
{
    writer << "@DATA" << endl;

    auto outputs = (double *)data.data;
    for (auto row = 0; row < data.rows; row++) 
    {
        for (auto column = 0; column < data.cols; column++) 
        {
            auto index = column + row * data.cols;
            if (column != 0) writer << ",";
            writer << setprecision(12) << outputs[index];        
        }
        writer << endl;
    }
}

//--------------------------------------------------
// Load Data
//--------------------------------------------------

/**
 * @brief Write the give data to disk
 * @param path The path that we are writing to
 * @return TrainData* The given set of training data
 */
TrainData * NeuralUtils::LoadData(const string& path) 
{
	auto fields = vector<string>(); Mat data = LoadARFF(path, fields);
	auto sourceData = (double *)data.data;

	Mat inputs = Mat_<float>::zeros(data.rows, fields.size()); auto inputData = (float *) inputs.data;
	Mat outputs = Mat_<float>::zeros(data.rows, 1); auto outputData = (float *) outputs.data;

	for (auto row = 0; row < data.rows; row++) 
	{
		for (auto column = 0; column < data.cols; column++)
		{
			auto index = column + row * data.cols;
			auto value = (float) sourceData[index];

			if (column < inputs.cols) inputData[column + row * inputs.cols] = value;
			else outputData[row] = value;
		}
	}

	return new TrainData(inputs, outputs);
}

/**
 * @brief Load an ARFF file from disk
 * @param path A path to the file that we are loading
 * @param fieldNames The names of the fields that we are loading
 * @return Mat Returns a Mat
 */
Mat NeuralUtils::LoadARFF(const string& path, vector<string>& fieldNames)
{
	// Open the file
	auto reader = ifstream(path);
	if (!reader.is_open()) throw runtime_error("Unable to open file: " + path);

	// defines a line
	auto line = string();

	// Read in the attributes
	while(true) 
	{
		getline(reader, line);
		if (!reader.good()) break;

		if (NVLib::StringUtils::StartsWith(line, "@DATA")) break;

		if (NVLib::StringUtils::StartsWith(line, "@ATTRIBUTE")) 
		{
			auto parts = vector<string>(); NVLib::StringUtils::Split(line, ' ', parts);
			if (parts.size() == 3 && !(parts.size() > 2 && parts[1] == "class"))
			{
				fieldNames.push_back(parts[1]);
			}
		}
	}

	// Read in the data
	Mat result;
	while (true) 
	{
		getline(reader, line);
		if (line == string()) break;

		auto parts = vector<string>(); NVLib::StringUtils::Split(line, ',', parts);
		if (parts.size() != fieldNames.size() + 1) throw runtime_error("The file has bad data records");

		auto record = vector<double>(); for (auto& part : parts) record.push_back(NVLib::StringUtils::String2Double(part));
		Mat row = Mat(record).t();
		result.push_back(row);
	}

	// Close the file
	reader.close();

	// Return the result
	return result;
}

//--------------------------------------------------
// Create Network
//--------------------------------------------------

/**
 * @brief Create the network that we are using
 * @param structure The structure of the given network
 * @return The resultant network as outputs
 */
Ptr<ml::ANN_MLP> NeuralUtils::CreateNetwork(const string structure, int inputCount , int outputCount) 
{
	// Break the parameter set
	auto parts = vector<string>(); NVLib::StringUtils::Split(structure, ',', parts);

	Mat_<int> layers(parts.size() + 2, 1);

	layers(0) = inputCount;

	for (auto i = 0; i < parts.size(); i++) 
	{
		auto count = NVLib::StringUtils::String2Int(parts[i]);
		layers(i + 1) = count;
	}

    layers(layers.rows - 1) = outputCount;

    auto result = ml::ANN_MLP::create();
    result->setLayerSizes(layers);
    result->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0, 0);
    result->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.0001));
    result->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
 
	return result;
}

//--------------------------------------------------
// Retrieve score
//--------------------------------------------------

/**
 * @brief Calculate the score
 * @param data The data that we are getting the score for
 * @param network The associated neural network
 * @return double The value that the score includes
 */
double NeuralUtils::GetScore(TrainData * data, Ptr<ml::ANN_MLP>& network) 
{
	Mat result; network->predict(data->GetInputs(), result, ml::StatModel::RAW_OUTPUT);

	auto score = 0.0; auto actual = (float *) result.data; auto expected = (float *) data->GetOutputs().data;
	for (auto row = 0; row < result.rows; row++) 
	{
		auto diff = abs(actual[row] - expected[row]);
		score += diff;
	}

	return score;
}