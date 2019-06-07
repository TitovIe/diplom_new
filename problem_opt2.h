
/**
 *  @file test_vars_constr_cost.h
 *
 *  @brief Example to generate a solver-independent formulation for the problem, taken
 *  from the IPOPT cpp_example.
 *
 *  The example problem to be solved is given as:
 *
 *      min_x f(x) = -(x1-2)^2
 *      s.t.
 *           0 = x0^2 + x1 - 1
 *           -1 <= x0 <= 1
 *
 * In this simple example we only use one set of variables, constraints and
 * cost. However, most real world problems have multiple different constraints
 * and also different variable sets representing different quantities. This
 * framework allows to define each set of variables or constraints absolutely
 * independently from another and correctly stitches them together to form the
 * final optimization problem.
 *
 * For a helpful graphical overview, see:
 * http://docs.ros.org/api/ifopt/html/group__ProblemFormulation.html
 */

#include <ifopt/variable_set.h>
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <vector>
#include <graph.h>

/*Оператор, реализующий сумму элементов вектора1 и вектора2 */
template <typename T>
vector<T> operator+ (const vector<T>& v1,
                     const vector<T>& v2){
    vector<T> v_new;
    for(size_t i = 0; i < v1.size(); i++) {
        v_new.push_back(v1[i] + v2[i]);
    }

    return v_new;
}

/*Оператор, реализующий разность элементов вектора1 и вектора2 */
template <typename T>
vector<T> operator- (const vector<T>& v1,
                     const vector<T>& v2){
    vector<T> v_new;
    for(size_t i = 0; i < v1.size(); i++) {
        v_new.push_back(v1[i] - v2[i]);
    }

    return v_new;
}

/*Оператор, реализующий умножение элементов вектора на число */
template <typename T>
vector<double> operator* (const vector<T>& v, int number){
    vector<double> v_new;
    for(const auto& i : v){
        v_new.push_back(i * number);
    }
    return v_new;
}

/*Оператор, реализующий деление элементов вектора на число */
template <typename T>
vector<double> operator/ (const vector<T>& v, int number) {
    vector<double> v_new;
    for(const auto& i : v){
        v_new.push_back(i / number);
    }
    return v_new;
}

namespace ifopt {
    using namespace Eigen;

    //Получаем текущий объект(в данном случае наш граф)
    Graph graph = GetClobalObject();
    
    //Здесь считаем значение функции S для каждого вектора
    pair<double, vector<double>> S;
    pair<double, vector<double>> Si_calc(VectorXd& Ji){
        double s_sum = 0;
        double Jij_sigma_sum = 0;
        double Si = 0;
        vector<double> Si_grad(Graph::N - 1);

        for (const auto& sample : graph.Get_m_samples()) {
            for (int j = 0, k = 0; j < Graph::N - 1; j++, k++) {
                if (k == global_counter)
                    k++;

                /*Считаем сумму по j: Sum(-Jij * sigmai * sigmaj)
                * для одной выборки */
                Jij_sigma_sum += -Ji(j)
                                 * sample[k]
                                 * sample[global_counter];
            }
            /*Считаем сумму для M выборок, чтобы найти среднее.
             * Также находим градиент от Si,
             * в цикле - сумму градиенов для М выборок */
            s_sum += exp(Jij_sigma_sum);

            for(int j = 0, k = 0; j < Graph::N - 1; j++, k++){
                if(k == global_counter)
                    k++;

                Si_grad[j] += sample[k]
                              * exp(Jij_sigma_sum);
            }

            Jij_sigma_sum = 0;
        }
        /*Находим средний градиент по M выборкам
         * а также среднее Si */
        Si_grad = Si_grad * (-1) / Graph::M;
        Si = s_sum / Graph::M;

        return {Si, Si_grad};
    }
    

    class ExVariables : public VariableSet {
    public:
        // Every variable set has a name, here "var_set1". this allows the constraints
        // and costs to define values and Jacobians specifically w.r.t this variable set.
        ExVariables() : ExVariables("var_set1") {};
        ExVariables(const std::string& name) : VariableSet(2 * (Graph::N - 1), name)
        {
            // the initial values where the NLP starts iterating from
            Ji_vector = graph.Get_Ji_vector()[global_counter];
            pi_vector = graph.Get_constraint_vector()[global_counter];
        }

        // Here is where you can transform the Eigen::Vector into whatever
        // internal representation of your variables you have (here two doubles, but
        // can also be complex classes such as splines, etc..
        void SetVariables(const VectorXd& x) override
        {
            for(size_t i = 0; i < Graph::N - 1; i++){
                Ji_vector[i] = x(i);
            }
            
            for(size_t i = Graph::N - 1, j = 0; i < 2 * (Graph::N - 1); i++, j++){
                pi_vector[j] = x(i);
            }
        };

        // Here is the reverse transformation from the internal representation to
        // to the Eigen::Vector
        VectorXd GetValues() const override
        {
            VectorXd v(2 * (Graph::N - 1));
            for(size_t i = 0; i < Graph::N - 1; i++){
                v[i] = Ji_vector[i];
            }

            for(size_t i = 0, j = Graph::N - 1; i < Graph::N - 1; i++, j++){
                v[j] = pi_vector[i];
            }
            
            return v;
        };

        // Each variable has an upper and lower bound set here
        VecBound GetBounds() const override
        {
            VecBound bounds(GetRows());
            for(size_t i = 0; i < 2 * (Graph::N - 1); i++)
                bounds.at(i) = NoBound;
            
            return bounds;
        }

    private:
        vector<double> Ji_vector;
        vector<double> pi_vector;
    };


    class ExConstraint : public ConstraintSet {
    public:
        ExConstraint() : ExConstraint("constraint1") {}

        // This constraint set just contains 1 constraint, however generally
        // each set can contain multiple related constraints.
        ExConstraint(const std::string& name) : ConstraintSet(2 * (Graph::N - 1), name) {}

        // The constraint value minus the constant value "1", moved to bounds.
        VectorXd GetValues() const override
        {
            VectorXd g(GetRows());
            VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();
            
            for(size_t i = 0, j = Graph::N - 1; i < Graph::N - 1; j++, i++){
                g(i) = x(i) - x(j);
            }

            for(size_t i = 0, j = Graph::N - 1; i < Graph::N - 1; j++, i++){
                g(j) = x(i) + x(j);
            }

            //g(0) = x(0);
            //g(1) = x(1);
            //g(2) = x(2);
                        
            return g;
        };

        // The only constraint in this set is an equality constraint to 1.
        // Constant values should always be put into GetBounds(), not GetValues().
        // For inequality constraints (<,>), use Bounds(x, inf) or Bounds(-inf, x).
        VecBound GetBounds() const override
        {
            VecBound b(GetRows());
            //VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();

            for(size_t i = 0; i < Graph::N - 1; i++){
                b.at(i) = Bounds(-inf, 0);
            }

            for(size_t i = Graph::N - 1; i < 2 * (Graph::N - 1); i++){
                b.at(i) = Bounds(0, inf);
            }
            //b.at(0) = Bounds(-abs(graph.Get_constraint_vector()[1][0]), abs(graph.Get_constraint_vector()[1][0]));
            //b.at(1) = Bounds(-abs(graph.Get_constraint_vector()[1][1]), abs(graph.Get_constraint_vector()[1][1]));
            //b.at(2) = Bounds(-abs(graph.Get_constraint_vector()[1][2]), abs(graph.Get_constraint_vector()[1][2]));
            
            return b;
        }

        // This function provides the first derivative of the constraints.
        // In case this is too difficult to write, you can also tell the solvers to
        // approximate the derivatives by finite differences and not overwrite this
        // function, e.g. in ipopt.cc::use_jacobian_approximation_ = true
        void FillJacobianBlock (std::string var_set, Jacobian& jac_block) const override
        {
            // must fill only that submatrix of the overall Jacobian that relates
            // to this constraint and "var_set1". even if more constraints or variables
            // classes are added, this submatrix will always start at row 0 and column 0,
            // thereby being independent from the overall problem.
            if (var_set == "var_set1") {
                VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();

                jac_block.coeffRef(0, 0) = 1;
                jac_block.coeffRef(0, 1) = 0;
                jac_block.coeffRef(0, 2) = 0;
                jac_block.coeffRef(0, 3) = -1;
                jac_block.coeffRef(0, 4) = 0;
                jac_block.coeffRef(0, 5) = 0;
                
                jac_block.coeffRef(1, 0) = 0;
                jac_block.coeffRef(1, 1) = 1;
                jac_block.coeffRef(1, 2) = 0;
                jac_block.coeffRef(1, 3) = 0;
                jac_block.coeffRef(1, 4) = -1;
                jac_block.coeffRef(1, 5) = 0;
                
                jac_block.coeffRef(2, 0) = 0;
                jac_block.coeffRef(2, 1) = 0;
                jac_block.coeffRef(2, 2) = 1;
                jac_block.coeffRef(2, 3) = 0;
                jac_block.coeffRef(2, 4) = 0;
                jac_block.coeffRef(2, 5) = -1;

                jac_block.coeffRef(3, 0) = 1;
                jac_block.coeffRef(3, 1) = 0;
                jac_block.coeffRef(3, 2) = 0;
                jac_block.coeffRef(3, 3) = 1;
                jac_block.coeffRef(3, 4) = 0;
                jac_block.coeffRef(3, 5) = 0;

                jac_block.coeffRef(4, 0) = 0;
                jac_block.coeffRef(4, 1) = 1;
                jac_block.coeffRef(4, 2) = 0;
                jac_block.coeffRef(4, 3) = 0;
                jac_block.coeffRef(4, 4) = 1;
                jac_block.coeffRef(4, 5) = 0;

                jac_block.coeffRef(5, 0) = 0;
                jac_block.coeffRef(5, 1) = 0;
                jac_block.coeffRef(5, 2) = 1;
                jac_block.coeffRef(5, 3) = 0;
                jac_block.coeffRef(5, 4) = 0;
                jac_block.coeffRef(5, 5) = 1;
            }
        }
    };


    class ExCost: public CostTerm {
    public:
        ExCost() : ExCost("cost_term1") {}
        ExCost(const std::string& name) : CostTerm(name) {}

        double GetCost() const override
        {
            VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();
            double func_value = 0;

            S = Si_calc(x);
            func_value += S.first;
            for(size_t i = Graph::N - 1; i < 2 * (Graph::N - 1); i++)
                func_value += x(i) * graph.lambda;
            
            return func_value;
        };

        void FillJacobianBlock (std::string var_set, Jacobian& jac) const override
        {
            if (var_set == "var_set1") {
                VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();

                for(size_t i = 0; i < Graph::N - 1; i++)
                    jac.coeffRef(0, i) = S.second[i];             // derivative of cost w.r.t x0

                for(size_t i = Graph::N - 1; i < 2 * (Graph::N - 1); i++)
                    jac.coeffRef(0, i) = graph.lambda;
            }
        }
    };

} // namespace opt