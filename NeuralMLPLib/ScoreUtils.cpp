//--------------------------------------------------
// Implementation of class ScoreUtils
//
// @author: Wild Boar
//
// @date: 2022-09-26
//--------------------------------------------------

#include "ScoreUtils.h"
using namespace NVL_AI;

//--------------------------------------------------
// Load
//--------------------------------------------------

/**
 * @brief Load an ARFF file from disk
 * @param path A path to the file that we are loading
 * @param fieldNames The names of the fields that we are loading
 * @return Mat Returns a Mat
 */
Mat ScoreUtils::LoadARFF(const string& path, vector<string>& fieldNames)
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