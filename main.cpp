#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>
#include "graph.h"
#include "problem_opt2.h"
//#include </home/titov/CoinIpopt/include/coin/IpIpoptApplication.hpp>

//using namespace Ipopt;

using namespace ifopt;

int main() {
    //mt19937 gen;
    //gen.seed(time(0));

    Graph graph = GetClobalObject();
    graph.Print_sigma_all();
    graph.Print_Ji();

    //Graph_reconst_calc(graph, 0.001);
    //graph.Print_Ji();
    
    //for(size_t i = 0; i < 2; i++) {
       // for (; global_counter < Graph::N; global_counter++) {
            Problem nlp;
            nlp.AddVariableSet(make_shared<ExVariables>());
            nlp.AddConstraintSet(make_shared<ExConstraint>());
            nlp.AddCostSet(make_shared<ExCost>());
            nlp.PrintCurrent();

            // 2. choose solver and options
            IpoptSolver ipopt;
            ipopt.SetOption("linear_solver", "mumps");
            ipopt.SetOption("jacobian_approximation", "exact");

            // 3 . solve
            ipopt.Solve(nlp);
            Eigen::VectorXd x = nlp.GetOptVariables()->GetValues();
            cout << x.transpose() << endl;
        //}
        //Jij_to_zero(graph);
        //graph.lambda = 0;
        //global_counter = 0;
    //}

    //graph.Print_Ji();
    // 4. test if solution correct
    //double eps = 1e-5; //double precision
    //assert(1.0-eps < x(0) && x(0) < 1.0+eps);
    //assert(0.0-eps < x(1) && x(1) < 0.0+eps);
    
    
    //graph.Print_graph();
    //cout << S_calculate(graph);
    //cout << Gradient();

    return 0;
}

