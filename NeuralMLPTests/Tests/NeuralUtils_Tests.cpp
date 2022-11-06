//--------------------------------------------------
// Unit Tests for NeuralUtils
//
// @author: Wild Boar
//
// @date: 2022-11-05
//--------------------------------------------------

#include <gtest/gtest.h>

#include <NVLib/FileUtils.h>

#include <NeuralMLPLib/NeuralUtils.h>

//--------------------------------------------------
// Test Helpers
//--------------------------------------------------

void Insert(Mat& matrix, int row, const vector<double>& values);

//--------------------------------------------------
// Test Methods
//--------------------------------------------------

/**
 * @brief Confirm that the data has been loaded correctly
 */
TEST(NeuralUtils_Test, test_data_load)
{
	// Create some test data
	Mat data = Mat_<double>::zeros(4, 3);
	Insert(data, 0, vector<double> { 0, 0, 0});
	Insert(data, 1, vector<double> { 0, 1, 1});
	Insert(data, 1, vector<double> { 1, 0, 1});
	Insert(data, 1, vector<double> { 1, 1, 0});

	// Write the test data to disk
	if (NVLib::FileUtils::Exists("test.arff")) NVLib::FileUtils::Remove("test.arff");
	NVL_AI::NeuralUtils::WriteData("test.arff", data);

	// Load the test data up again
	auto trainData = NVL_AI::NeuralUtils::LoadData("test.arff");

	// Confirm that the data has been loaded correctly
	ASSERT_EQ(trainData->GetInputs().cols, 2);
	ASSERT_EQ(trainData->GetInputs().rows, 4);
	ASSERT_EQ(trainData->GetOutputs().cols, 1);
	ASSERT_EQ(trainData->GetOutputs().rows, 4);

	auto input = (float *) trainData->GetInputs().data;
	auto output = (float *) trainData->GetOutputs().data;

	ASSERT_EQ(input[0], 0); ASSERT_EQ(input[1], 0); ASSERT_EQ(output[0], 0);
	ASSERT_EQ(input[2], 0); ASSERT_EQ(input[3], 1); ASSERT_EQ(output[1], 1);
	ASSERT_EQ(input[4], 1); ASSERT_EQ(input[5], 0); ASSERT_EQ(output[2], 1);
	ASSERT_EQ(input[6], 1); ASSERT_EQ(input[7], 1); ASSERT_EQ(output[3], 0);

	// Free the working memory
	delete trainData;
}

/**
 * @brief Confirm network initialization
 */
TEST(NeuralUtils_Test, confirm_network_initialization)
{
	// Get the network
	auto network = NVL_AI::NeuralUtils::CreateNetwork("3,3");
	Mat layerSizes = network->getLayerSizes();
	auto input = (int *) layerSizes.data;

	// Confirm that the network was correctly loaded
	ASSERT_EQ(layerSizes.rows, 4);
	ASSERT_EQ(layerSizes.cols, 1);

	ASSERT_EQ(input[0], 2);
	ASSERT_EQ(input[1], 3);
	ASSERT_EQ(input[2], 3);
	ASSERT_EQ(input[3], 1);
}

//--------------------------------------------------
// Helper Methods
//--------------------------------------------------

/**
 * @brief Inserts some values into the network
 * @param row The row that that we are setting the values to
 * @param matrix The matrix that we are putting the values into
 * @param values The values that we are inserting
 */
void Insert(Mat& matrix, int row, const vector<double>& values) 
{
	auto input = (double *) matrix.data;
	
	for (auto column = 0; column < matrix.cols; column++) 
	{
		auto index = column + row * matrix.cols;
		input[index] = values[column];
	}
}