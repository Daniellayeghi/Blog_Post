
#ifndef TEST_CPP_AUTODIFF_TEST_2_H
#define TEST_CPP_AUTODIFF_TEST_2_H

#include <functional>
#include "eigen3/Eigen/Core"
using namespace Eigen;
#include "autodiff/forward.hpp"
#include "autodiff/forward/eigen.hpp"
using namespace autodiff;


class CostFunction
{
public:
    using OBJ_FUNC_PTR = dual(*)(VectorXdual &x, VectorXdual &u);
    explicit CostFunction(OBJ_FUNC_PTR running, OBJ_FUNC_PTR terminal);
    void update_state(autodiff::VectorXdual& x, autodiff::VectorXdual& u);

    dual Lf();
    VectorXd Lf_x();
    VectorXd Lf_xx();
    VectorXd L_x();
    VectorXd L_u();
    VectorXd L_xx();
    VectorXd L_ux();

private:

    OBJ_FUNC_PTR _running_cost;
    OBJ_FUNC_PTR _terminal_cost;
    VectorXdual _u;
    VectorXdual _x;
};
#endif //TEST_CPP_AUTODIFF_TEST_2_H
