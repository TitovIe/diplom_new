#include "graph.h"

Graph::Graph(mt19937& gen){
    Gen_Ji_start_vector(gen);
    Gen_sigma_vector(gen);
}

void Graph::Gen_sigma_vector(mt19937 &gen) {
    MonteCarlo(sigmai_vector, gen);
}

/*Генерируем стартовые векторы Jij и pij*/
void Graph::Gen_Ji_start_vector(mt19937& gen){
    uniform_real_distribution<double>
            urd_right(Graph::alfa, Graph::beta);

    uniform_real_distribution<double>
            urd_left(-Graph::beta, -Graph::alfa);

    uniform_int_distribution<int> urd_false_true(0, 1);

    Ji_vector.reserve(Graph::N);
    constraint_vector.reserve(Graph::N);
    
    vector<double> Jij_start_vector(Graph::N - 1);
    vector<double> constraint_start_vector(Graph::N - 1);
    
    for(int k = 0; k < Graph::N; k++){
        Ji_vector.push_back(Jij_start_vector);
        constraint_vector.push_back(constraint_start_vector);
    }

    for(int i = 0; i < Graph::N; i++) {
        for (int j = i; j < Graph::N - 1; j++) {
            if (urd_false_true(gen) == 0) {
                Ji_vector[i][j] = urd_left(gen);
            } else Ji_vector[i][j] = urd_right(gen);;
            Ji_vector[j + 1][i] = Ji_vector[i][j];
            constraint_vector[i][j] = 1 / lambda * Ji_vector[i][j];
            constraint_vector[j + 1][i] = constraint_vector[i][j];
        }
    }
}

//Генерация M выборок спинов методом метрополиса
void MonteCarlo(vector<vector<int>>& samples, mt19937 &gen){

    uniform_int_distribution<int> urd_false_true(0, 1);
    uniform_real_distribution<double> probability(0, 1);
    uniform_int_distribution<int> choise_spin(0, Graph::N - 1);

    const int T = 273;
    const double k = 1.38 * pow(10, -23);
    const double beta = 1/k/T;

    const int number_step = 100;
    int number_spin;
    double E_prev, E_after, E_delta;

    // Данные, которые хотим получить
    vector<vector<double>> Jij_true;
    Jij_true.push_back({0.7, 0, 0});
    Jij_true.push_back({0.7, 0.41, -0.99});
    Jij_true.push_back({0, 0.41, 0.5});
    Jij_true.push_back({0, -0.99, 0.5});

//    Jij_true.push_back({0.7, 0, 0.5, 0, 0.6, -0.4, 0.59, 0.8});
//    Jij_true.push_back({0.7, 0.41, 0, -0.8, 0, 0.6, -0,5, 0.8});
//    Jij_true.push_back({0, 0.41, 0, -0.5, 0, 0, 0.5, 0});
//    Jij_true.push_back({0.5, 0, 0, 0.4, -0.46, 0, 0, -0.8});
//    Jij_true.push_back({0, -0.8, -0.5, 0.4, 0.9, -0.55, 0, -0.7});
//    Jij_true.push_back({0.6, 0, 0, -0.46, 0.9, 0, 0.9, 0});
//    Jij_true.push_back({-0.4, 0.6, 0, 0, -0.55, 0, 0, 0});
//    Jij_true.push_back({0.59, -0.5, 0.5, 0, 0, 0.9, 0, 0});
//    Jij_true.push_back({0.8, 0.8, 0, -0.8, -0.7, 0, 0, 0});
    
    // Начальная конфигурация рандомно заполняется
    vector<int> sigma_vector;
    for (int j = 0; j < Graph::N; j++) {
        if (urd_false_true(gen) == 0)
            sigma_vector.push_back(-1);
        else
            sigma_vector.push_back(1);
    }

    /*Генерируем M выборок. Каждая новая конфигурация - через 100 итераций.
     * Каждая итерацию выбираем какой то узел и с вероятностью 50% делаем следующие операции.
     * Считаем энергию системы с текущим значением спина. Переворачиваем его 
     * и снова считаем. Если deltaЕ < 0, оставляем перевернутым. В противном случае
     * с вероятностью ext(-beta * E_delta) оставляем перевернутым. */
    for(int j = 0; j < Graph::M; j++) {
        for (int i = 0; i < number_step; i++){
            if(probability(gen) < 0.5)
                continue;
            
            number_spin = choise_spin(gen);
            E_prev = Jij_sum_calc(Jij_true, sigma_vector);

            sigma_vector[number_spin] *= -1;
            E_after = Jij_sum_calc(Jij_true, sigma_vector);
            E_delta = E_after - E_prev;

            if(E_delta > 0){
                if(probability(gen) > exp(-beta * E_delta))
                    sigma_vector[number_spin] *= -1;
            }
        }
        samples.push_back(sigma_vector);
    }
}

//Считаем энергию системы
double Jij_sum_calc(const vector<vector<double>>& Jij_vector,
        const vector<int>& sigma_vector){
    double Jij_sum = 0;
    for(int i = 0; i < Jij_vector.size(); i++){
        for(int j = i, k = j + 1; j < Jij_vector[i].size(); j++, k++){
            Jij_sum += -Jij_vector[i][j]
                        * sigma_vector[i]
                        * sigma_vector[k];
        }
    }
    return Jij_sum;
}


vector<vector<int>>& Graph::Get_m_samples() {
    return sigmai_vector;
}

vector<vector<double>>& Graph::Get_Ji_vector() {
    return Ji_vector;
}

void Graph::Print_Ji() {
    for(const auto& i : Get_Ji_vector()){
        cout << i << endl;
    }
    cout << endl;
}

void Graph::Print_sigma_all() {
    for(const auto& sample : sigmai_vector)
        Print_sigma_sample(sample);
}

void Graph::Print_sigma_sample(const vector<int>& sample) {
    for(const auto& i : sample) {
        cout << i << " ";
    }
    cout << endl;
}

vector<vector<double>>& Graph::Get_constraint_vector() {
    return constraint_vector;
}

ostream& operator << (ostream& os, const vector<double>& v) {
    for (const auto &j : v) {
        os.width(10);
        os << j << " ";
    }
    return os;
}

Graph& GetClobalObject()
{
    static mt19937 gen(time(0));
    static Graph graph(gen);
    return graph;
}

/*Приводим к нулю Jij такие,которые < alfa/2 */
void Jij_to_zero(Graph& g){
    for(int i = 0; i < Graph::N; i++){
        for(int j = i; j < Graph::N - 1; j++){
            g.Get_Ji_vector()[i][j] = (g.Get_Ji_vector()[i][j]
                                       + g.Get_Ji_vector()[j + 1][i]) / 2;

            if(g.Get_Ji_vector()[i][j] > -Graph::alfa / 2
               && g.Get_Ji_vector()[i][j] < Graph::alfa / 2)
                g.Get_Ji_vector()[i][j] = 0;

            g.Get_Ji_vector()[j + 1][i] = g.Get_Ji_vector()[i][j];
        }
    }
}
