#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <random>

using namespace std;


double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double predict(const vector<double>& x, const vector<double>& weights) {
    double z = weights[0]; // откл
    for (size_t i = 0; i < x.size(); ++i) {
        z += weights[i + 1] * x[i];
    }
    return sigmoid(z);
}

void train(vector<vector<double>>& X, vector<int>& y, vector<double>& weights,
    double alpha, int epochs, double lambda) {
    size_t m = X.size();
    size_t n = X[0].size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        vector<double> gradients(n + 1, 0.0);

        for (size_t i = 0; i < m; ++i) {
            double prediction = predict(X[i], weights);
            double error = prediction - y[i];
            gradients[0] += error; // bias
            for (size_t j = 0; j < n; ++j) {
                gradients[j + 1] += error * X[i][j];
            }
        }

        weights[0] -= alpha * gradients[0] / m;
        for (size_t j = 1; j < weights.size(); ++j) {
            weights[j] -= alpha * (gradients[j] / m + lambda * weights[j]);
        }

        if (epoch % 100 == 0) {
            double loss = 0.0;
            for (size_t i = 0; i < m; ++i) {
                double pred = predict(X[i], weights);
                loss += -y[i] * log(pred + 1e-15) - (1 - y[i]) * log(1 - pred + 1e-15);
            }

            double reg = 0.0;
            for (size_t j = 1; j < weights.size(); ++j) {
                reg += weights[j] * weights[j];
            }
        }
    }
}

void loadData(const string& filename, vector<vector<double>>& X, vector<int>& y) {
    ifstream file(filename);
    string line;
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> features;
        int label = 0;

        for (int col = 0; getline(ss, value, ','); ++col) {
            if (col == 8) label = stoi(value);
            else features.push_back(value.empty() ? 0.0 : stod(value));
        }

        if (features.size() == 8) {
            X.push_back(features);
            y.push_back(label);
        }
    }
}

void accuracy(const vector<vector<double>>& X, const vector<int>& y, const vector<double>& weights) {
    int tp = 0, tn = 0;
    size_t m = X.size();

    for (size_t i = 0; i < m; ++i) {
        double prob = predict(X[i], weights);
        int pred = prob >= 0.5 ? 1 : 0;

        if (pred == 1 && y[i] == 1) tp++;
        else if (pred == 0 && y[i] == 0) tn++;
    }

    double accuracy = (tp + tn) / static_cast<double>(m);

    cout << "Accuracy:  " << fixed << setprecision(4) << accuracy << endl;
}


int main() {
    vector<vector<double>> X;
    vector<int> y;

    loadData("data.csv", X, y);

    size_t n = X[0].size();
    vector<double> means(n, 0.0), stddevs(n, 1.0);

    for (size_t j = 0; j < n; ++j) {
        double sum = 0.0, sq_sum = 0.0;
        for (const auto& row : X) {
            sum += row[j];
            sq_sum += row[j] * row[j];
        }
        means[j] = sum / X.size();
        stddevs[j] = sqrt(sq_sum / X.size() - means[j] * means[j]);
        if (stddevs[j] == 0.0) stddevs[j] = 1.0;
    }

    for (auto& row : X) {
        for (size_t j = 0; j < n; ++j) {
            row[j] = (row[j] - means[j]) / stddevs[j];
        }
    }

    vector<double> weights(n + 1);
    default_random_engine eng;
    uniform_real_distribution<double> dist(-0.1, 0.1);
    for (auto& w : weights) w = dist(eng);

    double alpha = 0.05;
    int epochs = 10000;
    double lambda = 0.00001;

    train(X, y, weights, alpha, epochs, lambda);

    accuracy(X, y, weights);

    vector<double> newData = {7,107,74,0,0,29.6,0.254,31};

    for (size_t j = 0; j < n; ++j) {
        newData[j] = (newData[j] - means[j]) / stddevs[j];
    }

    double newProb = predict(newData, weights);
    int newPred = newProb >= 0.5 ? 1 : 0;

    cout << "\nПредсказание: " << newPred
              << ", Вероятность = " << fixed << setprecision(4) << newProb << endl;

    return 0;
}