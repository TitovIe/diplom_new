#pragma once

#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <random>
#include <ctime>
#include <iostream>

using namespace std;

static int global_counter = 0;

class Graph {
public:
    explicit Graph(mt19937& gen);
    vector<vector<int>>& Get_m_samples();
    vector<vector<double>>& Get_Ji_vector();
    vector<vector<double>>& Get_constraint_vector();
    void Gen_Ji_start_vector(mt19937& gen);
    void Gen_sigma_vector(mt19937& gen);

    void Print_sigma_all();
    void Print_sigma_sample(const vector<int>& sample);
    void Print_Ji();

    static const int M = 10000, N = 4, beta = 1;
    static constexpr double alfa = 0.4;
    const double eps = 0.05;
    const double c_lambda = 0.4;
    double lambda =
            c_lambda * sqrt(log(pow(N, 2) / eps) / M);
private:
    vector<vector<double>> Ji_vector;
    vector<vector<double>> constraint_vector;
    vector<vector<int>> sigmai_vector;
};

void MonteCarlo(vector<vector<int>>& samples, mt19937& gen);
double Jij_sum_calc(const vector<vector<double>>& Jij_vector,
        const vector<int>& sigma_vector);

ostream& operator << (ostream& os, const vector<double>& v);

Graph& GetClobalObject();
void Jij_to_zero(Graph& g);
