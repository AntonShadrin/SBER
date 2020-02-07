#include <iostream>
#include <fstream>
#include "full_graph.h"
#include <random>
#include <cmath>
#include <math.h>


int main()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-1.0, 0);
	int maxN = 10;
	const double pi = std::acos(-1);
	int sample_size = 10;
	std::ofstream fout("result.txt");
	fout << "sample size = " << sample_size << std::endl << std::endl;
	for (int n = 3; n <= maxN; ++n)
	{
		fout << "n = " << n << std::endl;
		double mean = 0.0;
		double mean_sqr = 0.0;
		double mean_aproximate = 0.0;
		double mean_minimal = 0.0;
		for (int j = 0; j < sample_size; j++)
		{
			std::vector<point> init_vector;
			for (int i = 0; i < n; ++i)
			{
				double x = abs(dis(gen));
				double y = abs(dis(gen));
				x = cos(2.0 * pi * x) * pow((-2.0)*log(y), 0.5);
				x = sin(2.0 * pi * x) * pow((-2.0)*log(y), 0.5);
				init_vector.push_back({ x,y });
			}
			FullGraph graph(init_vector);
			graph.CalculateDistance();
			double approximate = graph.GetApproximateWeightGraphPath();
			double minimal = graph.GetMinimalWeightGraphPath();
			mean += approximate - minimal;
			mean_aproximate += approximate;
			mean_minimal += minimal;
			mean_sqr += pow(approximate - minimal,2);
		}
		mean_aproximate /= double(sample_size);
		mean_minimal /= double(sample_size);
		mean /= double(sample_size);
		mean_sqr /= double(sample_size);
		if(mean_sqr!=0)
			mean_sqr = pow(mean_sqr, 0.5);
		fout << "mean_aproximate = " << mean_aproximate << std::endl;
		fout << "mean_minimal = " << mean_minimal << std::endl;
		fout << "mean = " << mean << std::endl;
		fout << "mean_sqr = " << mean_sqr << std::endl << std::endl;
	}

	return 0;
}